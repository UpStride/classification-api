import collections
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from .generic_model import GenericModel
import re
import itertools
import math
import numpy as np
from absl import logging
from six.moves import xrange

from tensorflow.python.eager import tape as tape_lib


GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'use_se',
    'local_pooling', 'condconv_num_experts', 'clip_projection_output',
    'blocks_args', 'fix_head_stem', 'grad_checkpoint'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type', 'fused_conv',
    'super_pixel', 'condconv'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

_DEFAULT_BLOCKS_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]

def round_filters(filters, global_params, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.width_coefficient
  divisor = global_params.depth_divisor
  min_depth = global_params.min_depth
  if skip or not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return int(new_filters)


def round_repeats(repeats, global_params, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_coefficient
  if skip or not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.initializers.variance_scaling uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random.normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.

  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


######
##
## UTILS
##
######
def _recompute_grad(f):
  """An eager-compatible version of recompute_grad.

  For f(*args, **kwargs), this supports gradients with respect to args or
  kwargs, but kwargs are currently only supported in eager-mode.
  Note that for keras layer and model objects, this is handled automatically.

  Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
  be able to access the member variables of that object, because `g` returns
  through the wrapper function `inner`.  When recomputing gradients through
  objects that inherit from keras, we suggest keeping a reference to the
  underlying object around for the purpose of accessing these variables.

  Args:
    f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.

  Returns:
   A function `g` that wraps `f`, but which recomputes `f` on the backwards
   pass of a gradient call.
  """

  @tf1.custom_gradient
  def inner(*args, **kwargs):
    """Inner function closure for calculating gradients."""
    current_var_scope = tf1.get_variable_scope()
    with tape_lib.stop_recording():
      result = f(*args, **kwargs)

    def grad_wrapper(*wrapper_args, **grad_kwargs):
      """Wrapper function to accomodate lack of kwargs in graph mode decorator."""

      @tf1.custom_gradient
      def inner_recompute_grad(*dresult):
        """Nested custom gradient function for computing grads in reverse and forward mode autodiff."""
        # Gradient calculation for reverse mode autodiff.
        variables = grad_kwargs.get('variables')
        with tf1.GradientTape() as t:
          id_args = tf1.nest.map_structure(tf1.identity, args)
          t.watch(id_args)
          if variables is not None:
            t.watch(variables)
          with tf1.control_dependencies(dresult):
            with tf1.variable_scope(current_var_scope):
              result = f(*id_args, **kwargs)
        kw_vars = []
        if variables is not None:
          kw_vars = list(variables)
        grads = t.gradient(
            result,
            list(id_args) + kw_vars,
            output_gradients=dresult,
            unconnected_gradients=tf1.UnconnectedGradients.ZERO)

        def transpose(*t_args, **t_kwargs):
          """Gradient function calculation for forward mode autodiff."""
          # Just throw an error since gradients / activations are not stored on
          # tape for recompute.
          raise NotImplementedError(
              'recompute_grad tried to transpose grad of {}. '
              'Consider not using recompute_grad in forward mode'
              'autodiff'.format(f.__name__))

        return (grads[:len(id_args)], grads[len(id_args):]), transpose

      return inner_recompute_grad(*wrapper_args)

    return result, grad_wrapper

  return inner


def recompute_grad(recompute=False):
  """Decorator determine whether use gradient checkpoint."""

  def _wrapper(f):
    if recompute:
      return _recompute_grad(f)
    return f

  return _wrapper


def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = inputs / survival_prob * binary_tensor
  return output


class Stem(tf.keras.layers.Layer):
  """Stem layer at the begining of the network."""

  def __init__(self, framework, factor, global_params, stem_filters, name=None):
    super().__init__(name=name)
    self._conv_stem = framework.Conv2D(
        filters=round_filters(stem_filters // factor, global_params,
                              global_params.fix_head_stem),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=global_params.data_format,
        use_bias=False)
    self._bn = framework.BatchNormalization(
        axis=(1 if global_params.data_format == 'channels_first' else -1),
        momentum=global_params.batch_norm_momentum,
        epsilon=global_params.batch_norm_epsilon)
    self._relu_fn = global_params.relu_fn or tf.nn.swish

  def call(self, inputs, training):
    return self._relu_fn(self._bn(self._conv_stem(inputs), training=training))


class Head(tf.keras.layers.Layer):
  """Head layer for network outputs."""

  def __init__(self, framework, factor, num_classes, global_params, name=None):
    super().__init__(name=name)

    self.endpoints = {}
    self._global_params = global_params

    self._conv_head = framework.Conv2D(
        filters=round_filters(1280 * num_classes // 1000, global_params, global_params.fix_head_stem),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=global_params.data_format,
        use_bias=False,
        name='conv2d')
    self._bn = framework.BatchNormalization(
        axis=(1 if global_params.data_format == 'channels_first' else -1),
        momentum=global_params.batch_norm_momentum,
        epsilon=global_params.batch_norm_epsilon)
    self._relu_fn = global_params.relu_fn or tf.nn.swish

    self._avg_pooling = framework.GlobalAveragePooling2D(
        data_format=global_params.data_format)
    if global_params.num_classes:
      self._fc = framework.Dense(
          global_params.num_classes,
          kernel_initializer=dense_kernel_initializer)
    else:
      self._fc = None

    if global_params.dropout_rate > 0:
      self._dropout = framework.Dropout(global_params.dropout_rate)
    else:
      self._dropout = None

    self.h_axis, self.w_axis = (
        [2, 3] if global_params.data_format == 'channels_first' else [1, 2])

  def call(self, inputs, training, pooled_features_only=False):
    """Call the layer."""
    outputs = self._relu_fn(
        self._bn(self._conv_head(inputs), training=training))
    self.endpoints['head_1x1'] = outputs

    if self._global_params.local_pooling:
      shape = outputs.get_shape().as_list()
      kernel_size = [1, shape[self.h_axis], shape[self.w_axis], 1]
      outputs = tf.nn.avg_pool(
          outputs, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
      self.endpoints['pooled_features'] = outputs
      if not pooled_features_only:
        if self._dropout:
          outputs = self._dropout(outputs, training=training)
        self.endpoints['global_pool'] = outputs
        if self._fc:
          outputs = tf.squeeze(outputs, [self.h_axis, self.w_axis])
          outputs = self._fc(outputs)
        self.endpoints['head'] = outputs
    else:
      outputs = self._avg_pooling(outputs)
      self.endpoints['pooled_features'] = outputs
      if not pooled_features_only:
        if self._dropout:
          outputs = self._dropout(outputs, training=training)
        self.endpoints['global_pool'] = outputs
        if self._fc:
          outputs = self._fc(outputs)
        self.endpoints['head'] = outputs
    return outputs


class SuperPixel(tf.keras.layers.Layer):
  """Super pixel layer."""

  def __init__(self, framework, factor, block_args, global_params, name=None):
    super().__init__(name=name)
    self._superpixel = framework.Conv2D(
        block_args.input_filters,
        kernel_size=[2, 2],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=global_params.data_format,
        use_bias=False,
        name='conv2d')
    self._bnsp = framework.BatchNormalization(
        axis=1 if global_params.data_format == 'channels_first' else -1,
        momentum=global_params.batch_norm_momentum,
        epsilon=global_params.batch_norm_epsilon,
        name='tpu_batch_normalization')
    self._relu_fn = global_params.relu_fn or tf.nn.swish

  def call(self, inputs, training):
    return self._relu_fn(self._bnsp(self._superpixel(inputs), training))


class SE(tf.keras.layers.Layer):
  """Squeeze-and-excitation layer."""

  def __init__(self, framework, factor, global_params, se_filters, output_filters, name=None):
    super().__init__(name=name)

    self._local_pooling = global_params.local_pooling
    self._data_format = global_params.data_format
    self._relu_fn = global_params.relu_fn or tf.nn.swish

    # Squeeze and Excitation layer.
    self._se_reduce = framework.Conv2D(
        se_filters // factor,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        name='conv2d')
    self._se_expand = framework.Conv2D(
        output_filters // factor,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        name='conv2d_1')

  def call(self, inputs):
    h_axis, w_axis = [2, 3] if self._data_format == 'channels_first' else [1, 2]
    if self._local_pooling:
      se_tensor = tf.nn.avg_pool(
          inputs,
          ksize=[1, inputs.shape[h_axis], inputs.shape[w_axis], 1],
          strides=[1, 1, 1, 1],
          padding='VALID')
    else:
      se_tensor = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
    logging.info('Built SE %s : %s', self.name, se_tensor.shape)
    return tf.sigmoid(se_tensor) * inputs


class MBConvBlock(tf.keras.layers.Layer):
  """A class of MBConv: Mobile Inverted Residual Bottleneck.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, framework, factor, block_args, global_params, name=None):
    """Initializes a MBConv block.

    Args:
      block_args: BlockArgs, arguments to create a Block.
      global_params: GlobalParams, a set of global parameters.
      name: layer name.
    """
    super().__init__(name=name)

    self._block_args = block_args
    self._global_params = global_params
    self._local_pooling = global_params.local_pooling
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    self._batch_norm = framework.BatchNormalization
    self._condconv_num_experts = global_params.condconv_num_experts
    self._data_format = global_params.data_format
    self._channel_axis = 1 if self._data_format == 'channels_first' else -1

    self._relu_fn = global_params.relu_fn or tf.nn.swish
    self._has_se = (
        global_params.use_se and self._block_args.se_ratio is not None and
        0 < self._block_args.se_ratio <= 1)

    self._clip_projection_output = global_params.clip_projection_output

    self.endpoints = None

    if self._block_args.condconv:
      raise ValueError('Condconv is not supported.')

    # Builds the block accordings to arguments.
    self._build(framework=framework, factor=factor)

  @property
  def block_args(self):
    return self._block_args

  def _build(self, framework, factor):
    """Builds block according to the arguments."""
    bid = itertools.count(0)
    get_bn_name = lambda: 'tpu_batch_normalization' + ('' if not next(
        bid) else '_' + str(next(bid) // 2))
    cid = itertools.count(0)
    get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
        next(cid) // 2))

    if self._block_args.super_pixel == 1:
      self.super_pixel = SuperPixel(
          self._block_args, self._global_params, name='super_pixel')
    else:
      self.super_pixel = None

    filters = self._block_args.input_filters * self._block_args.expand_ratio
    kernel_size = self._block_args.kernel_size

    if self._block_args.fused_conv:
      # Fused expansion phase. Called if using fused convolutions.
      self._fused_conv = framework.Conv2D(
          filters=filters // factor,
          kernel_size=[kernel_size, kernel_size],
          strides=self._block_args.strides,
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          data_format=self._data_format,
          use_bias=False,
          name=get_conv_name())
    else:
      # Expansion phase. Called if not using fused convolutions and expansion
      # phase is necessary.
      if self._block_args.expand_ratio != 1:
        self._expand_conv = framework.Conv2D(
            filters=filters // factor,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False,
            name=get_conv_name())
        self._bn0 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            name=get_bn_name())

      # Depth-wise convolution phase. Called if not using fused convolutions.
      self._depthwise_conv = framework.DepthwiseConv2D(
          kernel_size=[kernel_size, kernel_size],
          strides=self._block_args.strides,
          depthwise_initializer=conv_kernel_initializer,
          padding='same',
          data_format=self._data_format,
          use_bias=False,
          name='depthwise_conv2d')

    self._bn1 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        name=get_bn_name())

    if self._has_se:
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))
      self._se = SE(framework, factor,
          self._global_params, num_reduced_filters, filters, name='se')
    else:
      self._se = None

    # Output phase.
    filters = self._block_args.output_filters
    self._project_conv = framework.Conv2D(
        filters=filters // factor,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False,
        name=get_conv_name())
    self._bn2 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        name=get_bn_name())

  def call(self, inputs, training, survival_prob=None):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
    @recompute_grad(self._global_params.grad_checkpoint)
    def _call(inputs):
      logging.info('Block %s input shape: %s', self.name, inputs.shape)
      x = inputs

      # creates conv 2x2 kernel
      if self.super_pixel:
        x = self.super_pixel(x, training)
        logging.info('SuperPixel %s: %s', self.name, x.shape)

      if self._block_args.fused_conv:
        # If use fused mbconv, skip expansion and use regular conv.
        x = self._relu_fn(self._bn1(self._fused_conv(x), training=training))
        logging.info('Conv2D shape: %s', x.shape)
      else:
        # Otherwise, first apply expansion and then apply depthwise conv.
        if self._block_args.expand_ratio != 1:
          x = self._relu_fn(self._bn0(self._expand_conv(x), training=training))
          logging.info('Expand shape: %s', x.shape)

        x = self._relu_fn(self._bn1(self._depthwise_conv(x), training=training))
        logging.info('DWConv shape: %s', x.shape)

      if self._se:
        x = self._se(x)

      self.endpoints = {'expansion_output': x}

      x = self._bn2(self._project_conv(x), training=training)
      # Add identity so that quantization-aware training can insert quantization
      # ops correctly.
      x = tf.identity(x)
      if self._clip_projection_output:
        x = tf.clip_by_value(x, -6, 6)
      if self._block_args.id_skip:
        if all(
            s == 1 for s in self._block_args.strides
        ) and self._block_args.input_filters == self._block_args.output_filters:
          # Apply only if skip connection presents.
          if survival_prob:
            x = drop_connect(x, training, survival_prob)
          x = tf.add(x, inputs)
      logging.info('Project shape: %s', x.shape)
      return x

    return _call(inputs)


class MBConvBlockWithoutDepthwise(MBConvBlock):
  """MBConv-like block without depthwise convolution and squeeze-and-excite."""

  def _build(self, framework, factor):
    """Builds block according to the arguments."""
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    cid = itertools.count(0)
    get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
        next(cid) // 2))
    kernel_size = self._block_args.kernel_size
    if self._block_args.expand_ratio != 1:
      # Expansion phase:
      self._expand_conv = framework.Conv2D(
          filters // factor,
          kernel_size=[kernel_size, kernel_size],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=False,
          name=get_conv_name())
      self._bn0 = self._batch_norm(
          axis=self._channel_axis,
          momentum=self._batch_norm_momentum,
          epsilon=self._batch_norm_epsilon)

    # Output phase:
    filters = self._block_args.output_filters
    self._project_conv = framework.Conv2D(
        filters // factor,
        kernel_size=[1, 1],
        strides=self._block_args.strides,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        name=get_conv_name())
    self._bn1 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

  def call(self, inputs, training, survival_prob=None):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """

    @recompute_grad(self._global_params.grad_checkpoint)
    def _call(inputs):
      logging.info('Block %s  input shape: %s', self.name, inputs.shape)
      if self._block_args.expand_ratio != 1:
        x = self._relu_fn(
            self._bn0(self._expand_conv(inputs), training=training))
      else:
        x = inputs
      logging.info('Expand shape: %s', x.shape)

      self.endpoints = {'expansion_output': x}

      x = self._bn1(self._project_conv(x), training=training)
      # Add identity so that quantization-aware training can insert quantization
      # ops correctly.
      x = tf.identity(x)
      if self._clip_projection_output:
        x = tf.clip_by_value(x, -6, 6)

      if self._block_args.id_skip:
        if all(
            s == 1 for s in self._block_args.strides
        ) and self._block_args.input_filters == self._block_args.output_filters:
          # Apply only if skip connection presents.
          if survival_prob:
            x = drop_connect(x, training, survival_prob)
          x = tf.add(x, inputs)
      logging.info('Project shape: %s', x.shape)
      return x

    return _call(inputs)


class BlockDecoder(object):
  """Block Decoder for readability."""

  def _decode_block_string(self, block_string):
    """Gets a block through a string notation of arguments."""
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    if 's' not in options or len(options['s']) != 2:
      raise ValueError('Strides options should be a pair of integers.')

    return BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]),
                 int(options['s'][1])],
        conv_type=int(options['c']) if 'c' in options else 0,
        fused_conv=int(options['f']) if 'f' in options else 0,
        super_pixel=int(options['p']) if 'p' in options else 0,
        condconv=('cc' in block_string))

  def _encode_block_string(self, block):
    """Encodes a block to a string."""
    args = [
        'r%d' % block.num_repeat,
        'k%d' % block.kernel_size,
        's%d%d' % (block.strides[0], block.strides[1]),
        'e%s' % block.expand_ratio,
        'i%d' % block.input_filters,
        'o%d' % block.output_filters,
        'c%d' % block.conv_type,
        'f%d' % block.fused_conv,
        'p%d' % block.super_pixel,
    ]
    if block.se_ratio > 0 and block.se_ratio <= 1:
      args.append('se%s' % block.se_ratio)
    if block.id_skip is False:
      args.append('noskip')
    if block.condconv:
      args.append('cc')
    return '_'.join(args)

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.

    Args:
      string_list: a list of strings, each string is a notation of block.

    Returns:
      A list of namedtuples to represent blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args

  def encode(self, blocks_args):
    """Encodes a list of Blocks to a list of strings.

    Args:
      blocks_args: A list of namedtuples to represent blocks arguments.
    Returns:
      a list of strings, each string is a notation of block.
    """
    block_strings = []
    for block in blocks_args:
      block_strings.append(self._encode_block_string(block))
    return block_strings


class EfficientNet(GenericModel):
  def __init__(self, blocks_args=None, global_params=None, *args, **kwargs):
    _, conversion_params, factor, _, num_classes, _, _ = args
    assert conversion_params["output_layer_before_up2tf"] == True, "COUCOU"

    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._global_params = global_params
    self._blocks_args = blocks_args
    self._relu_fn = global_params.relu_fn or tf.nn.swish
    self._fix_head_stem = global_params.fix_head_stem
    self.num_classes = num_classes
    self.factor = factor
    self.endpoints = None
    super().__init__(*args, **kwargs)

  def _get_conv_block(self, conv_type):
    conv_block_map = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}
    return conv_block_map[conv_type]

  def model(self):
    """Builds a model."""
    self._batch_norm = self.layers().BatchNormalization

    if self._global_params.data_format == 'channels_first':
      self.x = tf.transpose(self.x, [0, 3, 1, 2])
      tf.keras.backend.set_image_data_format('channels_first')

    self._blocks = []

    # Stem part.
    self._stem = Stem(framework=self.layers(), factor=self.factor, global_params=self._global_params, stem_filters=self._blocks_args[0].input_filters)

    # Builds blocks.
    block_id = itertools.count(0)
    block_name = lambda: 'blocks_%d' % next(block_id)
    for i, block_args in enumerate(self._blocks_args):
      assert block_args.num_repeat > 0
      assert block_args.super_pixel in [0, 1, 2]
      # Update block input and output filters based on depth multiplier.
      input_filters = round_filters(block_args.input_filters,
                                    self._global_params)

      output_filters = round_filters(block_args.output_filters,
                                     self._global_params)
      kernel_size = block_args.kernel_size
      if self._fix_head_stem and (i == 0 or i == len(self._blocks_args) - 1):
        repeats = block_args.num_repeat
      else:
        repeats = round_repeats(block_args.num_repeat, self._global_params)
      block_args = block_args._replace(
          input_filters=input_filters,
          output_filters=output_filters,
          num_repeat=repeats)

      # The first block needs to take care of stride and filter size increase.
      conv_block = self._get_conv_block(block_args.conv_type)
      if not block_args.super_pixel:  #  no super_pixel at all
        self._blocks.append(
            conv_block(framework=self.layers(), factor=self.factor, block_args=block_args, global_params=self._global_params, name=block_name()))
      else:
        # if superpixel, adjust filters, kernels, and strides.
        depth_factor = int(4 / block_args.strides[0] / block_args.strides[1])
        block_args = block_args._replace(
            input_filters=block_args.input_filters * depth_factor,
            output_filters=block_args.output_filters * depth_factor,
            kernel_size=((block_args.kernel_size + 1) //
                         2 if depth_factor > 1 else block_args.kernel_size))
        # if the first block has stride-2 and super_pixel trandformation
        if (block_args.strides[0] == 2 and block_args.strides[1] == 2):
          block_args = block_args._replace(strides=[1, 1])
          self._blocks.append(
              conv_block(framework=self.layers(), factor=self.factor, block_args=block_args, global_params=self._global_params, name=block_name()))
          block_args = block_args._replace(  # sp stops at stride-2
              super_pixel=0,
              input_filters=input_filters,
              output_filters=output_filters,
              kernel_size=kernel_size)
        elif block_args.super_pixel == 1:
          self._blocks.append(
              conv_block(framework=self.layers(), factor=self.factor, block_args=block_args, global_params=self._global_params, name=block_name()))
          block_args = block_args._replace(super_pixel=2)
        else:
          self._blocks.append(
              conv_block(framework=self.layers(), factor=self.factor, block_args=block_args, global_params=self._global_params, name=block_name()))
      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in xrange(block_args.num_repeat - 1):
        self._blocks.append(
            conv_block(framework=self.layers(), factor=self.factor, block_args=block_args, global_params=self._global_params, name=block_name()))

    # Head part.
    self._head = Head(framework=self.layers(), factor=self.factor, num_classes=self.num_classes, global_params=self._global_params)
    
    # NEW STUFF
    self.x = self._stem(self.x)
    for block in self._blocks:
      self.x = block(self.x)
    self.x = self._head(self.x)


class EfficientNetB0NCHW(EfficientNet):
  def __init__(self, *args, **kwargs):
    framework, conversion_params, factor, input_size, num_classes, _, _ = args
    assert tuple(input_size) == (224, 224, 3)

    global_params = GlobalParams(blocks_args=_DEFAULT_BLOCKS_ARGS,
                                 batch_norm_momentum=0.99,
                                 batch_norm_epsilon=1e-3,
                                 dropout_rate=0.2, #dropout_rate,
                                 survival_prob=None, #survival_prob,
                                 data_format='channels_first',
                                 num_classes=num_classes, #1000,
                                 width_coefficient=1.0, #width_coefficient,
                                 depth_coefficient=1.0, #depth_coefficient,
                                 depth_divisor=8,
                                 min_depth=None,
                                 relu_fn=tf.nn.swish,
                                 use_se=True,
                                 clip_projection_output=False)
    decoder = BlockDecoder()

    super().__init__(decoder.decode(global_params.blocks_args),
                     global_params,
                     *args,
                     **kwargs)