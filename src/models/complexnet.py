import tensorflow as tf
from .generic_model import GenericModel


class ComplexNet(GenericModel):
  def __init__(self, *args, **kwargs):
    self.channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    self.weight_regularizer = kwargs['weight_regularizer']
    self.bn_args = {
        "axis": self.channel_axis,
        "momentum": 0.9,
        "epsilon": 1e-04,
        # "center": False,
        # "scale": False
    }
    self.conv_args = {
        "padding": "same",
        "use_bias": False,
        "kernel_regularizer": self.weight_regularizer,
        "kernel_initializer": "he_ind"
    }
    super().__init__(*args, **kwargs)

  def residual_block(self, channels: int, downsample=False):
    layers = self.layers()
    x_init = self.x
    strides = (2, 2) if downsample else (1, 1)
    self.x = layers.BatchNormalizationC(**self.bn_args)(self.x)
    self.x = layers.Activation('relu')(self.x)
    self.x = layers.Conv2D(channels, 3, strides, **self.conv_args)(self.x)
    self.x = layers.BatchNormalizationC(**self.bn_args)(self.x)
    self.x = layers.Activation('relu')(self.x)
    self.x = layers.Conv2D(channels, 3, **self.conv_args)(self.x)
    if not downsample:
      self.x = layers.Add()([self.x, x_init])
    else:
      x_init = layers.Conv2D(channels, 1, 2, **self.conv_args)(x_init)
      self.x = layers.Concatenate(axis=self.channel_axis)([x_init, self.x])

  def learnVectorBlock(self):
    self.x = tf.keras.layers.BatchNormalization(**self.bn_args)(self.x)
    self.x = tf.keras.layers.Activation('relu')(self.x)
    self.x = tf.keras.layers.Convolution2D(3, self.lvb_kernel_size, kernel_initializer='he_normal', **self.conv_args)(self.x)
    self.x = tf.keras.layers.BatchNormalization(**self.bn_args)(self.x)
    self.x = tf.keras.layers.Activation('relu')(self.x)
    self.x = tf.keras.layers.Convolution2D(3, self.lvb_kernel_size, kernel_initializer='he_normal', **self.conv_args)(self.x)

  def model(self):
    n_channels = self.n_channels_type_0 // self.factor
    layers = self.layers()
    if layers == tf.keras.layers:
      print("real definition")
      r = self.x
      self.learnVectorBlock()
      self.x = tf.keras.layers.Concatenate()([r, self.x])
      self.conv_args['kernel_initializer'] = 'he_normal'

    self.x = self.layers().Conv2D(n_channels, 3, **self.conv_args)(self.x)
    self.x = self.layers().BatchNormalizationC(**self.bn_args)(self.x)
    self.x = self.layers().Activation('relu')(self.x)

    # First stage
    for i in range(self.n_blocks):  # -1 because the last one is a downsample
      self.residual_block(n_channels)
    self.residual_block(n_channels, True)

    # stage 2
    for i in range(self.n_blocks -  1):  # -1 because the last one is a downsample and one is removed (see paper)
      self.residual_block(n_channels * 2)
    self.residual_block(n_channels * 2, True)

    # stage 3
    for i in range(self.n_blocks -  1):  # -1 because the last one is a downsample and one is removed (see paper)
      self.residual_block(n_channels * 4)

    self.x = self.layers().GlobalAveragePooling2D()(self.x)

    # self.x = self.layers().Flatten()(self.x)

# Definition from the Quaternion Paper
class ShallowComplexNet(ComplexNet):
  def __init__(self, *args, **kwargs):
    self.conv_init = None
    self.n_blocks = 2
    self.n_channels_type_0 = 32
    self.lvb_kernel_size = 3
    super().__init__(*args, **kwargs)


class DeepComplexNet(ComplexNet):
  def __init__(self, *args, **kwargs):
    self.n_blocks = 11
    self.n_channels_type_0 = 32
    self.lvb_kernel_size = 3
    super().__init__(*args, **kwargs)


# definition from complex paper

# Wide and shallow definition
class WSComplexNetTF(ComplexNet):
  def __init__(self, *args, **kwargs):
    self.n_blocks = 14
    self.n_channels_type_0 = 18
    self.lvb_kernel_size = 1
    super().__init__(*args, **kwargs)


class WSComplexNetUpStride(ComplexNet):
  def __init__(self, *args, **kwargs):
    self.n_blocks = 16
    self.n_channels_type_0 = 12 * 2  # because 12 is the number of complex filter and we use factor 2
    self.lvb_kernel_size = 1
    super().__init__(*args, **kwargs)

# Deep and Narrow


class DNComplexNetTF(ComplexNet):
  def __init__(self, *args, **kwargs):
    self.n_blocks = 23
    self.n_channels_type_0 = 14
    self.lvb_kernel_size = 1
    super().__init__(*args, **kwargs)


class DNComplexNetUpStride(ComplexNet):
  def __init__(self, *args, **kwargs):
    self.n_blocks = 23
    self.n_channels_type_0 = 10 * 2  # because 12 is the number of complex filter and we use factor 2
    self.lvb_kernel_size = 1
    super().__init__(*args, **kwargs)


# In Between
class IBComplexNetTF(ComplexNet):
  def __init__(self, *args, **kwargs):
    self.n_blocks = 18
    self.n_channels_type_0 = 16
    self.lvb_kernel_size = 1
    super().__init__(*args, **kwargs)


class IBComplexNetUpStride(ComplexNet):
  def __init__(self, *args, **kwargs):
    self.n_blocks = 19
    self.n_channels_type_0 = 11 * 2  # because 12 is the number of complex filter and we use factor 2
    self.lvb_kernel_size = 1
    super().__init__(*args, **kwargs)
