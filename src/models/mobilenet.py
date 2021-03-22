import tensorflow as tf
from .generic_model import GenericModelBuilder

BATCHNORM_MOMENTUM = 0.9

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


class _MobileNetV2(GenericModelBuilder):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.last_block_output_shape = 3

  def _inverted_res_block(self, x, expansion, stride, alpha, filters, block_id):
    layers = self.layers  # we don't want to switch between tf and upstride in this block
    in_channels = self.last_block_output_shape

    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    inputs = x
    prefix = 'block_{}_'.format(block_id)

    if block_id:
      # Expand
      x = layers.Conv2D((expansion * in_channels), kernel_size=1, padding='same', use_bias=False, name=prefix + 'expand', kernel_regularizer=self.weight_regularizer)(x)
      x = layers.BatchNormalization(axis=self.channel_axis, epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'expand_BN')(x)
      x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
      prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
      x = layers.ZeroPadding2D(padding=correct_pad(x, 3, self.is_channels_first), name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same' if stride == 1 else 'valid',
                                    name=prefix + 'depthwise', depthwise_regularizer=self.weight_regularizer)(x)
    x = layers.BatchNormalization(axis=self.channel_axis, epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None,
                           name=prefix + 'project', kernel_regularizer=self.weight_regularizer)(x)
    x = layers.BatchNormalization(axis=self.channel_axis, epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
      x = layers.Add(name=prefix + 'add')([inputs, x])
    self.last_block_output_shape = pointwise_filters
    return x

  def model(self, x, alpha=1.0):
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
    weight_regularizer = self.weight_regularizer

    first_block_filters = _make_divisible(32 * alpha // self.factor, 8)
    x = self.layers.ZeroPadding2D(padding=correct_pad(x, 3, self.is_channels_first), name='Conv1_pad')(x)
    x = self.layers.Conv2D(first_block_filters, kernel_size=3, strides=self.first_conv_stride, padding='valid',
                                  use_bias=False, name='Conv1', kernel_regularizer=self.weight_regularizer)(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis, epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name='bn_Conv1')(x)
    x = self.layers.ReLU(6., name='Conv1_relu')(x)

    self.last_block_output_shape = first_block_filters

    block_id = 0
    for configuration in self.configurations:
      for i in range(configuration[1]):
        stride = configuration[2] if i == 0 else 1
        x = self._inverted_res_block(x, filters=configuration[0]//self.factor, alpha=alpha, stride=stride, expansion=configuration[3], block_id=block_id)
        block_id += 1

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
      last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
      last_block_filters = 1280
    last_block_filters = last_block_filters // self.factor

    x = self.layers.Conv2D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1', kernel_regularizer=self.weight_regularizer)(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis, epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name='Conv_1_bn')(x)
    x = self.layers.ReLU(6., name='out_relu')(x)

    x = self.layers.GlobalAveragePooling2D()(x)
    return x


class MobileNetV2(_MobileNetV2):
  def __init__(self, *args, **kwargs):
    # (channels, num_blocks, stride, expansion)
    self.first_conv_stride = 2
    self.configurations = [(16, 1, 1, 1),
                           (24, 2, 2, 6),
                           (32, 3, 2, 6),
                           (64, 4, 2, 6),
                           (96, 3, 1, 6),
                           (160, 3, 2, 6),
                           (320, 1, 1, 6)]
    super().__init__(*args, **kwargs)


class MobileNetV2Cifar10(_MobileNetV2):
  def __init__(self, *args, **kwargs):
    # (channels, num_blocks, stride, expansion)
    self.first_conv_stride = 1
    self.configurations = [(16, 1, 1, 1),
                           (24, 2, 1, 6),
                           (32, 3, 2, 6),
                           (64, 4, 2, 6),
                           (96, 3, 1, 6),
                           (160, 3, 2, 6),
                           (320, 1, 1, 6)]
    super().__init__(*args, **kwargs)

class MobileNetV2Cifar10_2(_MobileNetV2):
  def __init__(self, *args, **kwargs):
    # (channels, num_blocks, stride, expansion)
    self.first_conv_stride = 1
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

  def model(self, x):
    # (channels, num_blocks, stride, expansion)
    self.first_conv_stride = 1

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
    super().model(x)
