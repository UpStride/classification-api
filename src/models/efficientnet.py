"""EfficientNet models for Keras.
# Reference paper
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]
  (https://arxiv.org/abs/1905.11946) (ICML 2019)
# Reference implementation
- [TensorFlow]
  (https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

code from https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
adapted for upstride
"""
import os
import math
import tensorflow as tf
from copy import deepcopy
from .generic_model import GenericModelBuilder


def correct_pad(inputs, kernel_size, is_channels_first):
  """Returns a tuple for zero-padding for 2D convolution with downsampling.
  Args:
      input_size: An integer or tuple/list of 2 integers.
      kernel_size: An integer or tuple/list of 2 integers.
  Returns:
      A tuple.
  """
  if type(inputs) == list:
    inputs = inputs[0]
  input_size = inputs.shape[2:4] if is_channels_first else inputs.shape[1:3]
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  adjust = (1, 1) if input_size[0] is None else (1 - input_size[0] % 2, 1 - input_size[1] % 2)
  correct = (kernel_size[0] // 2, kernel_size[1] // 2)
  return ((correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1], correct[1]))

DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16, 'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24, 'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40, 'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80, 'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112, 'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192, 'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320, 'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def swish(x):
  """Swish activation function.
  # Arguments
      x: Input tensor.
  # Returns
      The Swish activation: `x * sigmoid(x)`.
  # References
      [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
  """
  return tf.nn.swish(x)


class EfficientNet(GenericModelBuilder):
  def __init__(self,
               width_coefficient,
               depth_coefficient,
               default_size,
               dropout_rate,
               drop_connect_rate=0.2,
               depth_divisor=8,
               activation_fn=swish,
               blocks_args=DEFAULT_BLOCKS_ARGS,
               pooling=None,
               *args, **kwargs
               ):
    """
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation_fn: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
    """
    self.width_coefficient = width_coefficient
    self.depth_coefficient = depth_coefficient
    self.default_size = default_size
    self.dropout_rate = dropout_rate
    self.drop_connect_rate = drop_connect_rate
    self.depth_divisor = depth_divisor
    self.activation_fn = activation_fn
    self.blocks_args = blocks_args
    super().__init__(*args, **kwargs)

  def round_filters(self, filters, divisor=None):
    """Round number of filters based on depth multiplier."""
    if divisor is None:
      divisor = self.depth_divisor
    filters *= self.width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
      new_filters += divisor
    return int(new_filters)

  def round_repeats(self, repeats):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(self.depth_coefficient * repeats))

  def block(self, inputs, activation_fn=swish, drop_rate=0., name='',
            filters_in=32, filters_out=16, kernel_size=3, strides=1,
            expand_ratio=1, se_ratio=0., id_skip=True):
    """A mobile inverted residual block.
    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
    # Returns
        output tensor for the block.
    """
    layers = self.layers # because don't want to change the framework in the middle of a block

    # Expansion phase
    filters = int(filters_in * expand_ratio) if expand_ratio == 1 else int(filters_in * expand_ratio / self.factor)
    if expand_ratio != 1:
      x = layers.Conv2D(filters, 1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name=name + 'expand_conv')(inputs)
      x = layers.BatchNormalization(axis=self.channel_axis, name=name + 'expand_bn')(x)
      x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
      x = inputs

    # Depthwise Convolution
    if strides == 2:
      x = layers.ZeroPadding2D(padding=correct_pad(x, kernel_size, self.is_channels_first), name=name + 'dwconv_pad')(x)
      conv_pad = 'valid'
    else:
      conv_pad = 'same'
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=conv_pad, use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER, name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=self.channel_axis, name=name + 'bn')(x)
    x = layers.Activation(activation_fn, name=name + 'activation')(x)
    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
      filters_se = max(1, int(filters_in * se_ratio)) if expand_ratio == 1 else max(1, int(filters_in * se_ratio / self.factor))
      se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
      if self.channel_axis == 1:
        se = layers.Reshape((filters, 1, 1), name=name + 'se_reshape')(se)
      else:
        se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
      se = layers.Conv2D(filters_se, 1, padding='same', activation=activation_fn, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'se_reduce')(se)
      se = layers.Conv2D(filters, 1, padding='same', activation='sigmoid', kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'se_expand')(se)
      x = layers.Multiply(name=name + 'se_excite')([x, se])

    # Output phase
    x = layers.Conv2D(filters_out/self.factor, 1, padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=self.channel_axis, name=name + 'project_bn')(x)
    if (id_skip is True and strides == 1 and filters_in == filters_out):
      if drop_rate > 0:
        x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
      x = layers.Add(name=name + 'add')([x, inputs])
    return x

  def model(self, x):
    # Build stem
    x = self.layers.ZeroPadding2D(padding=correct_pad(x, 3, self.is_channels_first))(x)
    x = self.layers.Conv2D(self.round_filters(32), 3,
                                  strides=2,
                                  padding='valid',
                                  use_bias=False,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name='stem_conv')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis, name='stem_bn')(x)
    x = self.layers.Activation(self.activation_fn, name='stem_activation')(x)

    # Build blocks
    blocks_args = deepcopy(self.blocks_args)
    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
      assert args['repeats'] > 0
      # Update block input and output filters based on depth multiplier.
      args['filters_in'] = self.round_filters(args['filters_in'])
      args['filters_out'] = self.round_filters(args['filters_out'])

      for j in range(self.round_repeats(args.pop('repeats'))):
        # The first block needs to take care of stride and filter size increase.
        if j > 0:
          args['strides'] = 1
          args['filters_in'] = args['filters_out']
        x = self.block(x, self.activation_fn, self.drop_connect_rate * b / blocks,
                            name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
        b += 1

    # Build top
    x = self.layers.Conv2D(self.round_filters(1280), 1,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name='top_conv')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis, name='top_bn')(x)
    x = self.layers.Activation(self.activation_fn, name='top_activation')(x)

    x = self.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if self.dropout_rate > 0:
      x = self.layers.Dropout(self.dropout_rate, name='top_dropout')(x)
    x = self.layers.Dense(self.num_classes,
                                 kernel_initializer=DENSE_KERNEL_INITIALIZER,
                                 name='probs')(x)
    return x


class EfficientNetB0(EfficientNet):
  def __init__(self, *args, **kwargs):
    super().__init__(1.0, 1.0, 224, 0.2, *args, **kwargs)


class EfficientNetB1(EfficientNet):
  def __init__(self, *args, **kwargs):
    super().__init__(1.0, 1.1, 240, 0.2, *args, **kwargs)


class EfficientNetB2(EfficientNet):
  def __init__(self, *args, **kwargs):
    super().__init__(1.1, 1.2, 260, 0.3, *args, **kwargs)


class EfficientNetB3(EfficientNet):
  def __init__(self, *args, **kwargs):
    super().__init__(1.2, 1.4, 300, 0.3, *args, **kwargs)


class EfficientNetB4(EfficientNet):
  def __init__(self, *args, **kwargs):
    super().__init__(1.4, 1.8, 380, 0.4, *args, **kwargs)


class EfficientNetB5(EfficientNet):
  def __init__(self, *args, **kwargs):
    super().__init__(1.6, 2.2, 456, 0.4, *args, **kwargs)


class EfficientNetB6(EfficientNet):
  def __init__(self, *args, **kwargs):
    super().__init__(1.8, 2.6, 528, 0.5, *args, **kwargs)


class EfficientNetB7(EfficientNet):
  def __init__(self, *args, **kwargs):
    super().__init__(2.0, 3.1, 600, 0.5, *args, **kwargs)
