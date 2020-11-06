import tensorflow as tf
from .generic_model import GenericModel

is_channel_fist = False

BATCHNORM_MOMENTUM = 0.9
weight_regularizer = tf.keras.regularizers.l2(l=0.0001)

# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py


def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


def correct_pad(inputs, kernel_size):
  """Returns a tuple for zero-padding for 2D convolution with downsampling.
  Args:
      input_size: An integer or tuple/list of 2 integers.
      kernel_size: An integer or tuple/list of 2 integers.
  Returns:
      A tuple.
  """
  if type(inputs) == list:
    inputs = inputs[0]
  input_size = inputs.shape[2:4] if is_channel_fist else inputs.shape[1:3]
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  adjust = (1, 1) if input_size[0] is None else (1 - input_size[0] % 2, 1 - input_size[1] % 2)
  correct = (kernel_size[0] // 2, kernel_size[1] // 2)
  return ((correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1], correct[1]))


class _MobileNetV2(GenericModel):
  def __init__(self, *args, **kwargs):
    self.last_block_output_shape = 3
    self.bn_axis = 1 if is_channel_fist else -1
    self.weight_regularizer = weight_regularizer
    super().__init__(*args, **kwargs)

  def _inverted_res_block(self, expansion, stride, alpha, filters, block_id):
    layers = self.layers()  # we don't want to switch between tf and upstride in this block
    in_channels = self.last_block_output_shape

    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    inputs = self.x
    prefix = 'block_{}_'.format(block_id)

    if block_id:
      # Expand
      self.x = layers.Conv2D((expansion * in_channels), kernel_size=1, padding='same', use_bias=False, name=prefix + 'expand', kernel_regularizer=weight_regularizer)(self.x)
      self.x = layers.BatchNormalization(axis=self.bn_axis, epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'expand_BN')(self.x)
      self.x = layers.ReLU(6., name=prefix + 'expand_relu')(self.x)
    else:
      prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
      self.x = layers.ZeroPadding2D(padding=correct_pad(self.x, 3), name=prefix + 'pad')(self.x)
    self.x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same' if stride == 1 else 'valid',
                                    name=prefix + 'depthwise', kernel_regularizer=weight_regularizer)(self.x)
    self.x = layers.BatchNormalization(axis=self.bn_axis, epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'depthwise_BN')(self.x)

    self.x = layers.ReLU(6., name=prefix + 'depthwise_relu')(self.x)

    # Project
    self.x = layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None,
                           name=prefix + 'project', kernel_regularizer=weight_regularizer)(self.x)
    self.x = layers.BatchNormalization(axis=self.bn_axis, epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'project_BN')(self.x)

    if in_channels == pointwise_filters and stride == 1:
      self.x = layers.Add(name=prefix + 'add')([inputs, self.x])
    self.last_block_output_shape = pointwise_filters

  def model(self, alpha=1.0):
    """Instantiates the MobileNetV2 architecture.
    Args:
        alpha: controls the width of the network. This is known as the
        width multiplier in the MobileNetV2 paper, but the name is kept for
        consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                are used at each layer.
    """
    if is_channel_fist:
      self.x = tf.transpose(self.x, [0, 3, 1, 2])
      tf.keras.backend.set_image_data_format('channels_first')

    first_block_filters = _make_divisible(32 * alpha // self.factor, 8)
    self.x = self.layers().ZeroPadding2D(padding=correct_pad(self.x, 3), name='Conv1_pad')(self.x)
    self.x = self.layers().Conv2D(first_block_filters, kernel_size=3, strides=self.fist_conv_stride, padding='valid',
                                  use_bias=False, name='Conv1', kernel_regularizer=weight_regularizer)(self.x)
    self.x = self.layers().BatchNormalization(axis=self.bn_axis, epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name='bn_Conv1')(self.x)
    self.x = self.layers().ReLU(6., name='Conv1_relu')(self.x)

    self.last_block_output_shape = first_block_filters

    block_id = 0
    for configuration in self.configurations:
      for i in range(configuration[1]):
        stride = configuration[2] if i == 0 else 1
        self._inverted_res_block(filters=configuration[0]//self.factor, alpha=alpha, stride=stride, expansion=configuration[3], block_id=block_id)
        block_id += 1

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
      last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
      last_block_filters = 1280
    last_block_filters = last_block_filters // self.factor

    self.x = self.layers().Conv2D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1', kernel_regularizer=weight_regularizer)(self.x)
    self.x = self.layers().BatchNormalization(axis=self.bn_axis, epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name='Conv_1_bn')(self.x)
    self.x = self.layers().ReLU(6., name='out_relu')(self.x)

    self.x = self.layers().GlobalAveragePooling2D()(self.x)

#    self.x = self.layers().Dense(self.label_dim, use_bias=True, name='Logits', kernel_regularizer=weight_regularizer)(self.x)


class MobileNetV2(_MobileNetV2):
  def __init__(self, *args, **kwargs):
    # (channels, num_blocks, stride, expansion)
    self.fist_conv_stride = 2
    self.configurations = [(16, 1, 1, 1),
                           (24, 2, 2, 6),
                           (32, 3, 2, 6),
                           (64, 4, 2, 6),
                           (96, 3, 1, 6),
                           (160, 3, 2, 6),
                           (320, 1, 1, 6)]
    super().__init__(*args, **kwargs)


class MobileNetV2NCHW(MobileNetV2):
  def __init__(self, *args, **kwargs):
    global is_channel_fist
    is_channel_fist = True
    super().__init__(*args, **kwargs)


class MobileNetV2Cifar10(_MobileNetV2):
  def __init__(self, *args, **kwargs):
    # (channels, num_blocks, stride, expansion)
    self.fist_conv_stride = 1
    self.configurations = [(16, 1, 1, 1),
                           (24, 2, 1, 6),
                           (32, 3, 2, 6),
                           (64, 4, 2, 6),
                           (96, 3, 1, 6),
                           (160, 3, 2, 6),
                           (320, 1, 1, 6)]
    super().__init__(*args, **kwargs)


class MobileNetV2Cifar10NCHW(MobileNetV2Cifar10):
  def __init__(self, *args, **kwargs):
    global is_channel_fist
    is_channel_fist = True
    super().__init__(*args, **kwargs)


class MobileNetV2Cifar10_2(_MobileNetV2):
  def __init__(self, *args, **kwargs):
    # (channels, num_blocks, stride, expansion)
    self.fist_conv_stride = 1
    self.configurations = [(16, 1, 1, 1),
                           (24, 2, 1, 6),
                           (32, 3, 1, 6),
                           (64, 4, 2, 6),
                           (96, 3, 1, 6),
                           (160, 3, 2, 6),
                           (320, 1, 1, 6)]
    super().__init__(*args, **kwargs)


class MobileNetV2Cifar10Hyper(_MobileNetV2):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def model(self):
    # (channels, num_blocks, stride, expansion)
    self.fist_conv_stride = 1

    # define 10 MobileNetv2 versions
    blocks_family = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 1, 1, 1],
        [1, 1, 1, 2, 2, 1, 1],
        [1, 1, 2, 2, 2, 1, 1],
        [1, 1, 2, 2, 2, 2, 1],
        [1, 1, 2, 3, 2, 2, 1],
        [1, 2, 2, 3, 2, 2, 1],
        [1, 2, 3, 3, 2, 2, 1],
        [1, 2, 3, 3, 3, 2, 1],
        [1, 2, 3, 3, 3, 3, 1],
        [1, 2, 3, 4, 3, 3, 1]  # Default config
    ]

    self.mobilenet_version = self.hp.Int('depth', min_value=0, max_value=10, step=1)
    block = blocks_family[self.mobilenet_version]
    self.configurations = [(16, block[0], 1, 1),
                           (24,  block[1], 1, 6),
                           (32,  block[2], 1, 6),
                           (64,  block[3], 2, 6),
                           (96,  block[4], 1, 6),
                           (160, block[5], 2, 6),
                           (320, block[6], 1, 6)]
    super().model()
