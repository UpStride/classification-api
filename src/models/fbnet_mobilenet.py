import tensorflow as tf

import yaml

from .generic_model import GenericModelBuilder
from .fbnetv2 import ChannelMasking
from .mobilenet import correct_pad

BATCHNORM_MOMENTUM = 0.9
arch_param_regularizer = tf.keras.regularizers.l2(l=0.0005)

class _FBNet_MobileNetV2(GenericModelBuilder):
  def __init__(self, *args, load_searched_arch: str = None, **kwargs):
    """the official implementation use tf.keras.backend.int_shape(x)[-1] to compute in_channels.
    But with upstride we can't, because if working with the cpp engine, the shape of the tensor is multiply by the length of the multivector.
    To bypass this issue, we can remember the output shape of the last block and use it to define in_channels
    At the begining of the model building, the shape shoud be 3 (number of channels)
    """
    self.last_block_output_shape = 3
    self.load_searched_arch = load_searched_arch
    
    if self.load_searched_arch:
      if tf.io.gfile.exists(self.load_searched_arch):
        with open(self.load_searched_arch, 'r') as f:
          self.model_def = yaml.safe_load(f)
      else:
        raise FileNotFoundError(f"{self.load_searched_arch} incorrect, check the path")
      assert all([k1 == k2 for k1, k2 in zip(self.model_def.keys(), self.mapping.keys())]), "keys are not the same"
      for (k1, v1), (k2, v2) in zip(self.model_def.items(), self.mapping.items()):
          if k1 == k2:
            self.mapping[k2][0] = v1

    # init of super class need to be called at the end of this init because it calls model(), so everything need to be ready before
    super().__init__(*args, **kwargs)

  def _inverted_res_block(self, x, filters, stride, expansion, name):
    """This block performs the Conv(expansion)-> DepthWiseConv -> Conv(Projection))

    Args:
        expansion (integer): Interger value to increase the channels from the previous layer
        stride (Int): Strides to be applied in the convolution
        filters (Int) or tuple(Int): total feature maps to be obtained at the end of the block or range (min, max, step) during arch search
        name (str): Indicates the block number and controls expansion or just depthwise separable convolution
    """
    layers = self.layers
    weight_regularizer = self.weight_regularizer
    in_channels = self.last_block_output_shape
    # If model definition file is not passed
    if not self.load_searched_arch:
      # get the max possible number of channels
      pointwise_conv_filters = filters[1]
    else:
      # get the number of channels defined in the file
      pointwise_conv_filters = filters

    # TODO Enable this to see if we get speed up (multiples of 8 is required for activating tensor cores for mixed precision training)
    # pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    inputs = x
    prefix = name


    # Expand
    x = layers.Conv2D((expansion * in_channels), kernel_size=1, padding='same', use_bias=False, name=prefix + 'expand', kernel_regularizer=weight_regularizer)(x)
    # if not self.load_searched_arch:
      # new_filter_range = [i * expansion for i in filters]
      # x = ChannelMasking(*new_filter_range, name=prefix + '_cm1', regularizer=arch_param_regularizer)(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'expand_BN')(x)
    x = layers.ReLU(6., name=prefix + 'expand_relu')(x)

    # Depthwise
    if stride == 2:
      x = layers.ZeroPadding2D(padding=correct_pad(x, 3, self.is_channels_first), name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same' if stride == 1 else 'valid',
                                    name=prefix + 'depthwise', depthwise_regularizer=weight_regularizer)(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'depthwise_BN')(x)
    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_conv_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project', kernel_regularizer=weight_regularizer)(x)
    if not self.load_searched_arch:
      x = ChannelMasking(*filters, name=prefix + '_savable', regularizer=arch_param_regularizer)(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_conv_filters and stride == 1:
      x = layers.Add(name=prefix + 'add')([inputs, x])
    self.last_block_output_shape = pointwise_conv_filters

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

    # first_block_filters = _make_divisible(16, 8)
    x = self.layers.ZeroPadding2D(padding=correct_pad(x, 3, self.is_channels_first), name='conv1_pad')(x)

    first_block_filters = self.mapping['conv2d_01']
    if not self.load_searched_arch:
      x = self.layers.Conv2D(first_block_filters[0][1], kernel_size=3, strides=2, padding='valid', use_bias=False, name='conv2d_01', kernel_regularizer=weight_regularizer)(x)
      x = ChannelMasking(*first_block_filters[0], name='conv2d_01_savable', regularizer=arch_param_regularizer)(x)
      self.last_block_output_shape = first_block_filters[0][1]
    else:
      x = self.layers.Conv2D(first_block_filters[0], kernel_size=3, strides=2, padding='valid', use_bias=False, name='conv2d_01', kernel_regularizer=weight_regularizer)(x)
      self.last_block_output_shape = first_block_filters[0]
    x = self.layers.BatchNormalization(epsilon=1e-3, momentum=BATCHNORM_MOMENTUM)(x)
    x = self.layers.ReLU(6.)(x)
    

    # Inverted residuals
    for k, v in self.mapping.items():
      if k.split('_')[0] == 'irb':  # ignore conv2d for now
        x = self._inverted_res_block(x, filters=v[0], stride=v[1], expansion=v[2], name=k)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    last_block_filters = 1984  # TODO try with 1280

    # TODO move this into the _conv_block once we planned to use the channel masking for the below
    x = self.layers.Conv2D(last_block_filters, kernel_size=1, use_bias=False, kernel_regularizer=weight_regularizer)(x)
    # if not self.load_searched_arch:
    #     x = layers.ChannelMasking(, 1984, )(x) # TODO test
    x = self.layers.BatchNormalization(epsilon=1e-3, momentum=BATCHNORM_MOMENTUM)(x)
    x = self.layers.ReLU(6., name='out_relu')(x)

    x = self.layers.GlobalAveragePooling2D()(x)
    # x = self.layers.Dense(self.label_dim, use_bias=True, name='Logits', kernel_regularizer=weight_regularizer)(x)
    return x



class FBNet_MobileNetV2CIFAR(_FBNet_MobileNetV2):
  def __init__(self, *args, **kwargs):
    self.mapping = {
        # filter_range,  Stride,  expansion
        'conv2d_01': [(8, 16, 4),  1,        1],
        'irb_01': [(12, 16, 4),    1,        1],
        'irb_02': [(16, 24, 4),    1,        6],
        'irb_03': [(16, 24, 4),    1,        6],
        'irb_04': [(16, 24, 4),    1,        6],
        'irb_05': [(16, 40, 8),    1,        6],
        'irb_06': [(16, 40, 8),    1,        6],
        'irb_07': [(16, 40, 8),    1,        6],
        'irb_08': [(48, 80, 8),    1,        6],
        'irb_09': [(48, 80, 8),    1,        6],
        'irb_10': [(48, 80, 8),   1,        6],
        'irb_11': [(72, 112, 8),  1,        6],
        'irb_12': [(72, 112, 8),  1,        6],
        'irb_13': [(72, 112, 8),  1,        6],
        'irb_14': [(112, 184, 8), 2,        6],
        'irb_15': [(112, 184, 8), 1,        6],
        'irb_16': [(112, 184, 8), 1,        6],
        'irb_17': [(112, 184, 8), 1,        6],
        # 'conv2d_2': [1984,         1,        1],
      }
    super().__init__(*args, **kwargs)


class FBNet_MobileNetV2CIFARUP(_FBNet_MobileNetV2):
  def __init__(self, *args, **kwargs):
    self.mapping = {
        # filter_range,  Stride,  expansion
        'conv2d_01': [(4, 16, 4),  1,        1],
        'irb_01': [(4, 8, 4),    1,        1],
        'irb_02': [(4, 12, 4),    1,        6],
        'irb_03': [(4, 12, 4),    1,        6],
        'irb_04': [(4, 16, 4),    2,        6],
        'irb_05': [(4, 16, 4),    1,        6],
        'irb_06': [(4, 16, 4),    1,        6],
        'irb_07': [(8, 32, 4),    2,        6],
        'irb_08': [(8, 32, 4),    1,        6],
        'irb_09': [(8, 32, 4),    1,        6],
        'irb_10': [(8, 32, 4),   1,        6],
        'irb_11': [(12, 48, 4),  1,        6],
        'irb_12': [(12, 48, 4),  1,        6],
        'irb_13': [(12, 48, 4),  1,        6],
        'irb_14': [(24, 80, 8), 2,        6],
        'irb_15': [(24, 80, 8), 1,        6],
        'irb_16': [(24, 80, 8), 1,        6],
        'irb_17': [(40, 160, 8), 1,        6],
        # 'conv2d_2': [1984,         1,        1],
      }
    super().__init__(*args, **kwargs)

class FBNet_MobileNetV2Imagenet(_FBNet_MobileNetV2):
  def __init__(self, *args, **kwargs):
    self.mapping = {
        # filter_range,  Stride,  expansion
        'conv2d_01': [(8, 16, 4),  2,        1],
        'irb_01': [(12, 16, 4),    1,        1],
        'irb_02': [(16, 24, 4),    2,        6],
        'irb_03': [(16, 24, 4),    1,        6],
        'irb_04': [(16, 24, 4),    1,        6],
        'irb_05': [(16, 40, 8),    2,        6],
        'irb_06': [(16, 40, 8),    1,        6],
        'irb_07': [(16, 40, 8),    1,        6],
        'irb_08': [(48, 80, 8),    2,        6],
        'irb_09': [(48, 80, 8),    1,        6],
        'irb_10': [(48, 80, 8),   1,        6],
        'irb_11': [(72, 112, 8),  1,        6],
        'irb_12': [(72, 112, 8),  1,        6],
        'irb_13': [(72, 112, 8),  1,        6],
        'irb_14': [(112, 184, 8), 2,        6],
        'irb_15': [(112, 184, 8), 1,        6],
        'irb_16': [(112, 184, 8), 1,        6],
        'irb_17': [(112, 184, 8), 1,        6],
        # 'conv2d_2': [1984,         1,        1],
      }
    super().__init__(*args, **kwargs)