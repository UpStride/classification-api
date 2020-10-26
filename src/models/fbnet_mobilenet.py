import tensorflow as tf

import yaml

from .generic_model import GenericModel
from .fbnetv2 import ChannelMasking
from .mobilenet import correct_pad
is_channel_fist = False

BATCHNORM_MOMENTUM = 0.9
weight_regularizer = tf.keras.regularizers.l2(l=0.0001)
arch_param_regularizer = tf.keras.regularizers.l2(l=0.0005)

class _FBNet_MobileNetV2(GenericModel):
  def __init__(self, *args, load_searched_arch: str = None, **kwargs):
    """the official implementation use tf.keras.backend.int_shape(self.x)[-1] to compute in_channels.
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

  def _inverted_res_block(self, filters, stride, expansion, name):
    """This block performs the Conv(expansion)-> DepthWiseConv -> Conv(Projection))

    Args:
        expansion (integer): Interger value to increase the channels from the previous layer
        stride (Int): Strides to be applied in the convolution
        filters (Int) or tuple(Int): total feature maps to be obtained at the end of the block or range (min, max, step) during arch search
        name (str): Indicates the block number and controls expansion or just depthwise separable convolution
    """
    layers = self.layers()  # we don't want to switch between tf and upstride in this block
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
    inputs = self.x
    prefix = name


    # Expand
    self.x = layers.Conv2D((expansion * in_channels), kernel_size=1, padding='same', use_bias=False, name=prefix + 'expand', kernel_regularizer=weight_regularizer)(self.x)
    # if not self.load_searched_arch:
      # new_filter_range = [i * expansion for i in filters]
      # self.x = ChannelMasking(*new_filter_range, name=prefix + '_cm1', regularizer=arch_param_regularizer)(self.x)
    self.x = layers.BatchNormalization(epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'expand_BN')(self.x)
    self.x = layers.ReLU(6., name=prefix + 'expand_relu')(self.x)

    # Depthwise
    if stride == 2:
      self.x = layers.ZeroPadding2D(padding=correct_pad(self.x, 3), name=prefix + 'pad')(self.x)
    self.x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same' if stride == 1 else 'valid',
                                    name=prefix + 'depthwise', kernel_regularizer=weight_regularizer)(self.x)
    self.x = layers.BatchNormalization(epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'depthwise_BN')(self.x)
    self.x = layers.ReLU(6., name=prefix + 'depthwise_relu')(self.x)

    # Project
    self.x = layers.Conv2D(pointwise_conv_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project', kernel_regularizer=weight_regularizer)(self.x)
    if not self.load_searched_arch:
      self.x = ChannelMasking(*filters, name=prefix + '_savable', regularizer=arch_param_regularizer)(self.x)
    self.x = layers.BatchNormalization(epsilon=1e-3, momentum=BATCHNORM_MOMENTUM, name=prefix + 'project_BN')(self.x)

    if in_channels == pointwise_conv_filters and stride == 1:
      self.x = layers.Add(name=prefix + 'add')([inputs, self.x])
    self.last_block_output_shape = pointwise_conv_filters

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

    # first_block_filters = _make_divisible(16, 8)
    self.x = self.layers().ZeroPadding2D(padding=correct_pad(self.x, 3), name='conv1_pad')(self.x)

    first_block_filters = self.mapping['conv2d_01']
    if not self.load_searched_arch:
      self.x = self.layers().Conv2D(first_block_filters[0][1], kernel_size=3, strides=2, padding='valid', use_bias=False, name='conv2d_01', kernel_regularizer=weight_regularizer)(self.x)
      self.x = ChannelMasking(*first_block_filters[0], name='conv2d_01_savable', regularizer=arch_param_regularizer)(self.x)
      self.last_block_output_shape = first_block_filters[0][1]
    else:
      self.x = self.layers().Conv2D(first_block_filters[0], kernel_size=3, strides=2, padding='valid', use_bias=False, name='conv2d_01', kernel_regularizer=weight_regularizer)(self.x)
      self.last_block_output_shape = first_block_filters[0]
    self.x = self.layers().BatchNormalization(epsilon=1e-3, momentum=BATCHNORM_MOMENTUM)(self.x)
    self.x = self.layers().ReLU(6.)(self.x)
    

    # Inverted residuals
    for k, v in self.mapping.items():
      if k.split('_')[0] == 'irb':  # ignore conv2d for now
        self._inverted_res_block(filters=v[0], stride=v[1], expansion=v[2], name=k)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    last_block_filters = 1984  # TODO try with 1280

    # TODO move this into the _conv_block once we planned to use the channel masking for the below
    self.x = self.layers().Conv2D(last_block_filters, kernel_size=1, use_bias=False, kernel_regularizer=weight_regularizer)(self.x)
    # if not self.load_searched_arch:
    #     self.x = layers.ChannelMasking(, 1984, )(self.x) # TODO test
    self.x = self.layers().BatchNormalization(epsilon=1e-3, momentum=BATCHNORM_MOMENTUM)(self.x)
    self.x = self.layers().ReLU(6., name='out_relu')(self.x)

    self.x = self.layers().GlobalAveragePooling2D()(self.x)
    self.x = self.layers().Dense(self.label_dim, use_bias=True, name='Logits', kernel_regularizer=weight_regularizer)(self.x)



class FBNet_MobileNetV2NCHW(_FBNet_MobileNetV2):
  def __init__(self, *args, **kwargs):
    global is_channel_fist
    is_channel_fist = True
    super().__init__(*args, **kwargs)


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