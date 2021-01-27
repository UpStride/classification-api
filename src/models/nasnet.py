
"""NASNet-A models for Keras
NASNet refers to Neural Architecture Search Network, a family of models
that were designed automatically by learning the model architectures
directly on the dataset of interest.
Here we consider NASNet-A, the highest performance model that was found
for the CIFAR-10 dataset, and then extended to ImageNet 2012 dataset,
obtaining state of the art performance on CIFAR-10 and ImageNet 2012.
Only the NASNet-A models, and their respective weights, which are suited
for ImageNet 2012 are provided.
Performance on ImageNet 2012 reported on the paper:
------------------------------------------------------------------------------------
      Architecture       | Top-1 Acc | Top-5 Acc |  Multiply-Adds |  Params (M)
------------------------------------------------------------------------------------
|   NASNet-A (4 @ 1056)  |   74.0 %  |   91.6 %  |       564 M    |     5.3        |
|   NASNet-A (6 @ 4032)  |   82.7 %  |   96.2 %  |      23.8 B    |    88.9        |
------------------------------------------------------------------------------------

Results on Cifar10 reported on the paper:
-----------------------------------------------------------------
      Architecture                  | Error       |  Params (M)
------------------------------------------------------------------
|   NASNet-A (6 @ 768)              |   3.41 %    |    3.3M      |
|   NASNet-A (6 @ 768) + cutout     |   2.65 %    |    3.3M      |
------------------------------------------------------------------

"""
import warnings
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf
from .generic_model import GenericModelBuilder


def correct_pad(backend, inputs, kernel_size):
  """Returns a tuple for zero-padding for 2D convolution with downsampling.
  # Arguments
      input_size: An integer or tuple/list of 2 integers.
      kernel_size: An integer or tuple/list of 2 integers.
  # Returns
      A tuple.
  """
  img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
  input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)

  if input_size[0] is None:
    adjust = (1, 1)
  else:
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

  correct = (kernel_size[0] // 2, kernel_size[1] // 2)

  return ((correct[0] - adjust[0], correct[0]),
          (correct[1] - adjust[1], correct[1]))


class NASNet(GenericModelBuilder):
  def __init__(self, *args, **kwargs):
    self.backend = tf.keras.backend
    self.keras_utils = tf.keras.utils
    super().__init__(*args, **kwargs)

  def _separable_conv_block(self, ip, filters, kernel_size=(3, 3), strides=(1, 1), weight_decay=5e-5, id=None):
    '''Adds 2 blocks of [relu-separable conv-batchnorm]
    # Arguments:
        ip: input tensor
        filters: number of output filters per layer
        kernel_size: kernel size of separable convolutions
        strides: strided convolution for downsampling
        weight_decay: l2 regularization weight
        id: string id
    # Returns:
        a Keras tensor
    '''

    layers = self.layers
    channel_dim = 1 if self.backend.image_data_format() == 'channels_first' else -1

    with self.backend.name_scope('separable_conv_block_%s' % id):
      x = layers.Activation('relu')(ip)
      x = layers.SeparableConv2D(filters, kernel_size, strides=strides, name='separable_conv_1_%s' % id,
                                 padding='same', use_bias=False, kernel_initializer='he_normal',
                                 kernel_regularizer=l2(weight_decay))(x)
      x = layers.BatchNormalization(axis=channel_dim, momentum=self._BN_DECAY, epsilon=self._BN_EPSILON,
                                    name="separable_conv_1_bn_%s" % (id))(x)
      x = layers.Activation('relu')(x)
      x = layers.SeparableConv2D(filters, kernel_size, name='separable_conv_2_%s' % id,
                                 padding='same', use_bias=False, kernel_initializer='he_normal',
                                 kernel_regularizer=l2(weight_decay))(x)
      x = layers.BatchNormalization(axis=channel_dim, momentum=self._BN_DECAY, epsilon=self._BN_EPSILON,
                                    name="separable_conv_2_bn_%s" % (id))(x)
    return x

  def _adjust_block(self, p, ip, filters, weight_decay=5e-5, id=None):
    '''
    Adjusts the input `p` to match the shape of the `input`
    or situations where the output number of filters needs to
    be changed
    # Arguments:
        p: input tensor which needs to be modified
        ip: input tensor whose shape needs to be matched
        filters: number of output filters to be matched
        weight_decay: l2 regularization weight
        id: string id
    # Returns:
        an adjusted Keras tensor
    '''

    layers = self.layers

    channel_dim = 1 if self.backend.image_data_format() == 'channels_first' else -1
    img_dim = 2 if self.backend.image_data_format() == 'channels_first' else -2

    if type(ip) == list:
      ip_shape = self.backend.int_shape(ip[0])
    else:
      ip_shape = self.backend.int_shape(ip)

    if p is not None:
      if type(p) == list:
        p_shape = self.backend.int_shape(p[0])
      else:
        p_shape = self.backend.int_shape(p)

    with self.backend.name_scope('adjust_block'):
      if p is None:
        p = ip
      elif p_shape[img_dim] != ip_shape[img_dim]:
        with self.backend.name_scope('adjust_reduction_block_%s' % id):
          p = layers.Activation('relu', name='adjust_relu_1_%s' % id)(p)

          p1 = layers.AveragePooling2D((1, 1), strides=(2, 2), padding='valid',
                                       name='adjust_avg_pool_1_%s' % id)(p)
          p1 = layers.Conv2D(filters // 2, (1, 1), padding='same', use_bias=False,
                             kernel_regularizer=l2(weight_decay),
                             name='adjust_conv_1_%s' % id, kernel_initializer='he_normal')(p1)

          p2 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
          p2 = layers.Cropping2D(cropping=((1, 0), (1, 0)))(p2)
          p2 = layers.AveragePooling2D((1, 1), strides=(2, 2), padding='valid',
                                       name='adjust_avg_pool_2_%s' % id)(p2)
          p2 = layers.Conv2D(filters // 2, (1, 1), padding='same', use_bias=False,
                             kernel_regularizer=l2(weight_decay),
                             name='adjust_conv_2_%s' % id, kernel_initializer='he_normal')(p2)

          p = layers.Concatenate(axis=channel_dim)([p1, p2])
          p = layers.BatchNormalization(axis=channel_dim, momentum=self._BN_DECAY, epsilon=self._BN_EPSILON,
                                        name='adjust_bn_%s' % id)(p)
      elif p_shape[channel_dim] != filters:
        with self.backend.name_scope('adjust_projection_block_%s' % id):
          p = layers.Activation('relu')(p)
          p = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same',
                            name='adjust_conv_projection_%s' % id,
                            use_bias=False, kernel_regularizer=l2(weight_decay),
                            kernel_initializer='he_normal')(p)
          p = layers.BatchNormalization(axis=channel_dim, momentum=self._BN_DECAY, epsilon=self._BN_EPSILON,
                                        name='adjust_bn_%s' % id)(p)
    return p

  def _normal_A(self, ip, p, filters, weight_decay=5e-5, id=None):
    '''Adds a Normal cell for NASNet-A (Fig. 4 in the paper)
    # Arguments:
        ip: input tensor `x`
        p: input tensor `p`
        filters: number of output filters
        weight_decay: l2 regularization weight
        id: string id
    # Returns:
        a Keras tensor
    '''

    layers = self.layers

    channel_dim = 1 if self.backend.image_data_format() == 'channels_first' else -1

    with self.backend.name_scope('normal_A_block_%s' % id):
      p = self._adjust_block(p, ip, filters, weight_decay, id)

      h = layers.Activation('relu')(ip)
      h = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='normal_conv_1_%s' % id,
                        use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(h)
      h = layers.BatchNormalization(axis=channel_dim, momentum=self._BN_DECAY, epsilon=self._BN_EPSILON,
                                    name='normal_bn_1_%s' % id)(h)

      with self.backend.name_scope('block_1'):
        x1_1 = self._separable_conv_block(h, filters, kernel_size=(5, 5), weight_decay=weight_decay,
                                          id='normal_left1_%s' % id)
        x1_2 = self._separable_conv_block(p, filters, weight_decay=weight_decay, id='normal_right1_%s' % id)
        x1 = layers.Add(name='normal_add_1_%s' % id)([x1_1, x1_2])

      with self.backend.name_scope('block_2'):
        x2_1 = self._separable_conv_block(p, filters, (5, 5), weight_decay=weight_decay, id='normal_left2_%s' % id)
        x2_2 = self._separable_conv_block(p, filters, (3, 3), weight_decay=weight_decay, id='normal_right2_%s' % id)
        x2 = layers.Add(name='normal_add_2_%s' % id)([x2_1, x2_2])

      with self.backend.name_scope('block_3'):
        x3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left3_%s' % (id))(h)
        x3 = layers.Add(name='normal_add_3_%s' % id)([x3, p])

      with self.backend.name_scope('block_4'):
        x4_1 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left4_%s' % (id))(p)
        x4_2 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_right4_%s' % (id))(
            p)
        x4 = layers.Add(name='normal_add_4_%s' % id)([x4_1, x4_2])

      with self.backend.name_scope('block_5'):
        x5 = self._separable_conv_block(h, filters, weight_decay=weight_decay, id='normal_left5_%s' % id)
        x5 = layers.Add(name='normal_add_5_%s' % id)([x5, h])

      x = layers.Concatenate(axis=channel_dim, name='normal_concat_%s' % id)([p, x1, x2, x3, x4, x5])
    return x, ip

  def _reduction_A(self, ip, p, filters, weight_decay=5e-5, id=None):
    '''Adds a Reduction cell for NASNet-A (Fig. 4 in the paper)
    # Arguments:
        ip: input tensor `x`
        p: input tensor `p`
        filters: number of output filters
        weight_decay: l2 regularization weight
        id: string id
    # Returns:
        a Keras tensor
    '''
    """"""

    layers = self.layers
    channel_dim = 1 if self.backend.image_data_format() == 'channels_first' else -1

    with self.backend.name_scope('reduction_A_block_%s' % id):
      p = self._adjust_block(p, ip, filters, weight_decay, id)

      h = layers.Activation('relu')(ip)
      h = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='reduction_conv_1_%s' % id,
                        use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(h)
      h = layers.BatchNormalization(axis=channel_dim, momentum=self._BN_DECAY, epsilon=self._BN_EPSILON,
                                    name='reduction_bn_1_%s' % id)(h)

      with self.backend.name_scope('block_1'):
        x1_1 = self._separable_conv_block(h, filters, (5, 5), strides=(2, 2), weight_decay=weight_decay,
                                          id='reduction_left1_%s' % id)
        x1_2 = self._separable_conv_block(p, filters, (7, 7), strides=(2, 2), weight_decay=weight_decay,
                                          id='reduction_1_%s' % id)
        x1 = layers.Add(name='reduction_add_1_%s' % id)([x1_1, x1_2])

      with self.backend.name_scope('block_2'):
        x2_1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_left2_%s' % id)(h)
        x2_2 = self._separable_conv_block(p, filters, (7, 7), strides=(2, 2), weight_decay=weight_decay,
                                          id='reduction_right2_%s' % id)
        x2 = layers.Add(name='reduction_add_2_%s' % id)([x2_1, x2_2])

      with self.backend.name_scope('block_3'):
        x3_1 = layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_left3_%s' % id)(
            h)
        x3_2 = self._separable_conv_block(p, filters, (5, 5), strides=(2, 2), weight_decay=weight_decay,
                                          id='reduction_right3_%s' % id)
        x3 = layers.Add(name='reduction_add3_%s' % id)([x3_1, x3_2])

      with self.backend.name_scope('block_4'):
        x4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='reduction_left4_%s' % id)(x1)
        x4 = layers.Add()([x2, x4])

      with self.backend.name_scope('block_5'):
        x5_1 = self._separable_conv_block(x1, filters, (3, 3), weight_decay=weight_decay,
                                          id='reduction_left4_%s' % id)
        x5_2 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_right5_%s' % id)(h)
        x5 = layers.Add(name='reduction_add4_%s' % id)([x5_1, x5_2])

      x = layers.Concatenate(axis=channel_dim, name='reduction_concat_%s' % id)([x2, x3, x4, x5])
      return x, ip

  def _add_auxiliary_head(self, x, classes, weight_decay, pooling, include_top):
    '''Adds an auxiliary head for training the model
    From section A.7 "Training of ImageNet models" of the paper, all NASNet models are
    trained using an auxiliary classifier around 2/3 of the depth of the network, with
    a loss weight of 0.4
    # Arguments
        x: input tensor
        classes: number of output classes
        weight_decay: l2 regularization weight
    # Returns
        a keras Tensor
    '''

    layers = self.layers

    img_height = 1 if self.backend.image_data_format() == 'channels_last' else 2
    img_width = 2 if self.backend.image_data_format() == 'channels_last' else 3
    channel_axis = 1 if self.backend.image_data_format() == 'channels_first' else -1

    with self.backend.name_scope('auxiliary_branch'):
      auxiliary_x = layers.Activation('relu')(x)
      auxiliary_x = layers.AveragePooling2D((5, 5), strides=(3, 3), padding='valid', name='aux_pool')(auxiliary_x)
      auxiliary_x = layers.Conv2D(128, (1, 1), padding='same', use_bias=False, name='aux_conv_projection',
                                  kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
          auxiliary_x)
      auxiliary_x = layers.BatchNormalization(axis=channel_axis, momentum=self._BN_DECAY, epsilon=self._BN_EPSILON,
                                              name='aux_bn_projection')(auxiliary_x)
      auxiliary_x = layers.Activation('relu')(auxiliary_x)

      auxiliary_x_shape = self.backend.int_shape(auxiliary_x)
      auxiliary_x = layers.Conv2D(768, (auxiliary_x_shape[img_height], auxiliary_x_shape[img_width]),
                                  padding='valid', use_bias=False, kernel_initializer='he_normal',
                                  kernel_regularizer=l2(weight_decay), name='aux_conv_reduction')(auxiliary_x)
      auxiliary_x = layers.BatchNormalization(axis=channel_axis, momentum=self._BN_DECAY, epsilon=self._BN_EPSILON,
                                              name='aux_bn_reduction')(auxiliary_x)
      auxiliary_x = layers.Activation('relu')(auxiliary_x)

      if include_top:
        auxiliary_x = layers.Flatten()(auxiliary_x)
        auxiliary_x = layers.Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay),
                                   name='aux_predictions')(auxiliary_x)
      else:
        if pooling == 'avg':
          auxiliary_x = layers.GlobalAveragePooling2D()(auxiliary_x)
        elif pooling == 'max':
          auxiliary_x = layers.GlobalMaxPooling2D()(auxiliary_x)

    return auxiliary_x

  def model(self, x):
    """Instantiates a NASNet architecture.
    Note that only TensorFlow is supported for now,
    therefore it only works with the data format
    `image_data_format='channels_last'` in your Keras config
    at `~/.keras/keras.json`.
    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(331, 331, 3)` for NASNetLarge or
            `(224, 224, 3)` for NASNetMobile
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        penultimate_filters: number of filters in the penultimate layer.
            NASNet models use the notation `NASNet (N @ P)`, where:
                -   N is the number of blocks
                -   P is the number of penultimate filters
        nb_blocks: number of repeated blocks of the NASNet model.
            NASNet models use the notation `NASNet (N @ P)`, where:
                -   N is the number of blocks
                -   P is the number of penultimate filters
        stem_filters: number of filters in the initial stem block
        skip_reduction: Whether to skip the reduction step at the tail
            end of the network. Set to `True` for CIFAR models.
        skip_reduction_layer_input: Determines whether to skip the reduction layers
            when calculating the previous layer to connect to.
        use_auxiliary_branch: Whether to use the auxiliary branch during
            training or evaluation.
        filters_multiplier: controls the width of the network.
            - If `filters_multiplier` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `filters_multiplier` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `filters_multiplier` = 1, default number of filters from the paper
                 are used at each layer.
        dropout: dropout rate
        weight_decay: l2 regularization weight
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        default_size: specifies the default image size of the model
        with_upstride: training network with Upstride API if value is 1 else train with tensorflow
        factor: { 'upstride.type1': 1, 'upstride.type2' : 4 , 'upstride.type3  8' }
                Our underlying mathematics requires us to use the factor in order to
                have the same total number of floating points.
                Note: you might see a difference in the total network parameters when factor 4 or 8 is used.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """

    layers = self.layers

    if self.backend.image_data_format() != 'channels_last':
      warnings.warn('The NASNet family of models is only available '
                    'for the input data format "channels_last" '
                    '(width, height, channels). '
                    'However your settings specify the default '
                    'data format "channels_first" (channels, width, height).'
                    ' You should set `image_data_format="channels_last"` '
                    'in your Keras config located at ~/.keras/keras.json. '
                    'The model being returned right now will expect inputs '
                    'to follow the "channels_last" data format.')
      self.backend.set_image_data_format('channels_last')
      old_data_format = 'channels_first'
    else:
      old_data_format = None

    assert self.penultimate_filters % 24 == 0, "`penultimate_filters` needs to be divisible " \
        "by 24."

    penultimate_filters = self.penultimate_filters // self.factor
    stem_filters = self.stem_filters // self.factor

    channel_dim = 1 if self.backend.image_data_format() == 'channels_first' else -1
    filters = penultimate_filters // 24

    # Avoid odd number of filters to escape dimension mismatch
    if filters % 2 != 0:
      filters += 1

    if self.initial_reduction:
      x = layers.Conv2D(stem_filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False, name='stem_conv1',
                             kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(x)
    else:
      x = layers.Conv2D(stem_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='stem_conv1',
                             kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(x)

    x = layers.BatchNormalization(axis=channel_dim, momentum=self._BN_DECAY, epsilon=self._BN_EPSILON,
                                       name='stem_bn1')(x)

    p = None
    if self.initial_reduction:  # imagenet / mobile mode
      x, p = self._reduction_A(x, p, filters // (self.filters_multiplier ** 2), self.weight_decay, id='stem_1')
      x, p = self._reduction_A(x, p, filters // self.filters_multiplier, self.weight_decay, id='stem_2')

    for i in range(self.nb_blocks):
      x, p = self._normal_A(x, p, filters, self.weight_decay, id='%d' % (i))

    x, p0 = self._reduction_A(x, p, filters * self.filters_multiplier, self.weight_decay, id='reduce_%d' % (self.nb_blocks))

    p = p0 if not self.skip_reduction_layer_input else p

    for i in range(self.nb_blocks):
      x, p = self._normal_A(x, p, filters * self.filters_multiplier, self.weight_decay, id='%d' % (self.nb_blocks + i + 1))

    auxiliary_x = None
    if not self.initial_reduction:  # imagenet / mobile mode
      if self.use_auxiliary_branch:
        auxiliary_x = self._add_auxiliary_head(x, self.label_dim, self.weight_decay, self.pooling, self.include_top)

    x, p0 = self._reduction_A(x, p, filters * self.filters_multiplier ** 2, self.weight_decay, id='reduce_%d' % (2 * self.nb_blocks))

    if self.initial_reduction:  # CIFAR mode
      if self.use_auxiliary_branch:
        auxiliary_x = self._add_auxiliary_head(x, self.label_dim, self.weight_decay, self.pooling, self.include_top)

    p = p0 if not self.skip_reduction_layer_input else p

    for i in range(self.nb_blocks):
      x, p = self._normal_A(x, p, filters * self.filters_multiplier ** 2, self.weight_decay, id='%d' % (2 * self.nb_blocks + i + 1))

    x = layers.Activation('relu')(x)

    if self.include_top:
      x = layers.GlobalAveragePooling2D()(x)
      x = layers.Dropout(self.dropout)(x)
      x = layers.Dense(self.label_dim, kernel_regularizer=l2(self.weight_decay), name='logit')(x)
    else:
      if self.pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
      elif self.pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)

    # NASNet with auxiliary
    if self.use_auxiliary_branch:
      self.auxiliary_x = auxiliary_x

    if old_data_format:
      self.backend.set_image_data_format(old_data_format)
    return x


class NASNetLarge(NASNet):
  def __init__(self, *args, **kwargs):
    """Instantiates a NASNet architecture in CIFAR mode.
    # Instance attributes
        use_auxiliary_branch: Whether to use the auxiliary branch during
            training or evaluation.
        dropout: dropout rate
        weight_decay: l2 regularization weight
        include_top: whether to include the fully-connected
            layer at the top of the network.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
    """
    self._BN_DECAY = 0.9997
    self._BN_EPSILON = 1e-3
    self.penultimate_filters = 4032
    self.nb_blocks = 6
    self.stem_filters = 96
    self.initial_reduction = True
    self.skip_reduction_layer_input = False
    self.use_auxiliary_branch = False
    self.filters_multiplier = 2
    self.dropout = 0.5
    self.weight_decay = 5e-5
    self.include_top = True
    self.pooling = None
    super().__init__(*args, **kwargs)


class NASNetMobile(NASNet):
  def __init__(self, *args, **kwargs):
    """Instantiates a NASNet architecture in CIFAR mode.
    # Instance attributes
        use_auxiliary_branch: Whether to use the auxiliary branch during
            training or evaluation.
        dropout: dropout rate
        weight_decay: l2 regularization weight
        include_top: whether to include the fully-connected
            layer at the top of the network.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
    """
    self._BN_DECAY = 0.9997
    self._BN_EPSILON = 1e-3
    self.penultimate_filters = 1056
    self.nb_blocks = 4
    self.stem_filters = 32
    self.initial_reduction = True
    self.skip_reduction_layer_input = False
    self.use_auxiliary_branch = False
    self.filters_multiplier = 2
    self.dropout = 0.5
    self.weight_decay = 4e-5
    self.include_top = True
    self.pooling = None
    super().__init__(*args, **kwargs)


class NASNetCIFAR(NASNet):
  def __init__(self, *args, **kwargs):
    """Instantiates a NASNet architecture in CIFAR mode.
    # Instance attributes
        use_auxiliary_branch: Whether to use the auxiliary branch during
            training or evaluation.
        dropout: dropout rate
        weight_decay: l2 regularization weight
        include_top: whether to include the fully-connected
            layer at the top of the network.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
    """
    self._BN_DECAY = 0.9
    self._BN_EPSILON = 1e-5
    self.penultimate_filters = 768
    self.nb_blocks = 6
    self.stem_filters = 32
    self.initial_reduction = False
    self.skip_reduction_layer_input = False
    self.use_auxiliary_branch = False
    self.filters_multiplier = 2
    self.dropout = 0.0
    self.weight_decay = 5e-4
    self.include_top = True
    self.pooling = None
    super().__init__(*args, **kwargs)
