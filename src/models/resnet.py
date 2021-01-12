import tensorflow as tf
from .generic_model import GenericModel


weight_init = tf.keras.initializers.VarianceScaling()

is_channel_fist = False


class ResNet(GenericModel):
  def __init__(self, *args, **kwargs):
    super(ResNet, self).__init__(*args, **kwargs)

  def get_residual_layer(self):
    n_to_residual = {
        10: [1, 1, 1, 1],
        12: [1, 1, 2, 1],
        14: [1, 2, 2, 1],
        16: [2, 2, 2, 1],
        18: [2, 2, 2, 2],
        20: [2, 2, 3, 2],
        22: [2, 3, 3, 2],
        24: [2, 3, 4, 2],
        26: [2, 3, 5, 2],
        28: [2, 3, 6, 2],
        30: [2, 4, 6, 2],
        32: [3, 4, 6, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }
    return n_to_residual[self.res_n]

  def model(self):
    if is_channel_fist:
      self.x = tf.transpose(self.x, [0, 3, 1, 2])
      tf.keras.backend.set_image_data_format('channels_first')

    if self.res_n < 50:
      residual_block = self.resblock
    else:
      residual_block = self.bottle_resblock
    residual_list = self.get_residual_layer()
    ch = 64
    weight_regularizer = self.weight_regularizer
    self.x = self.layers().Conv2D(int(ch/self.factor), 7, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, padding="same", name='conv')(self.x)
    self.x = self.layers().MaxPooling2D(pool_size=3, strides=2)(self.x)
    for i in range(residual_list[0]):
      residual_block(channels=int(ch/self.factor), downsample=False, block_name='resblock0_' + str(i))
    # block 1
    residual_block(channels=int(ch/self.factor) * 2, downsample=True, block_name='resblock1_0')
    for i in range(1, residual_list[1]):
      residual_block(channels=int(ch/self.factor) * 2, downsample=False, block_name='resblock1_' + str(i))
    # block 2
    residual_block(channels=int(ch/self.factor) * 4, downsample=True, block_name='resblock2_0')
    for i in range(1, residual_list[2]):
      residual_block(channels=int(ch/self.factor) * 4, downsample=False, block_name='resblock2_' + str(i))
    # block 3
    residual_block(channels=int(ch/self.factor) * 8, downsample=True, block_name='resblock_3_0')
    for i in range(1, residual_list[3]):
      residual_block(channels=int(ch/self.factor) * 8, downsample=False, block_name='resblock_3_' + str(i))
    # block 4
    self.x = self.layers().BatchNormalization(name='batch_norm_last')(self.x)
    self.x = self.layers().Activation('relu', name='relu_last')(self.x)
    self.x = self.layers().GlobalAveragePooling2D()(self.x)
    self.x = self.layers().Dense(units=self.label_dim, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=True,  name='logit')(self.x)

  def resblock(self, channels, use_bias=True, downsample=False, block_name='resblock'):
    layers = self.layers()
    weight_regularizer = self.weight_regularizer
    x_init = self.x
    self.x = layers.BatchNormalization(name=block_name + '/batch_norm_0')(self.x)
    self.x = layers.Activation('relu', name=block_name + '/relu_0')(self.x)
    if downsample:
      self.x = layers.Conv2D(channels, 3, 2, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                             use_bias=use_bias, padding='same', name=block_name + '/conv_0')(self.x)
      x_init = layers.Conv2D(channels, 1, 2, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                             use_bias=use_bias, padding='same', name=block_name + '/conv_init')(x_init)
    else:
      self.x = layers.Conv2D(channels, 3, 1, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                             use_bias=use_bias, padding='same', name=block_name + '/conv_0')(self.x)
    self.x = layers.BatchNormalization(name=block_name + '/batch_norm_1')(self.x)
    self.x = layers.Activation('relu', name=block_name + '/relu_1')(self.x)
    self.x = layers.Conv2D(channels, 3, 1, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                           use_bias=use_bias, padding='same', name=block_name + '/conv_1')(self.x)
    self.x = layers.Add()([self.x, x_init])

  def bottle_resblock(self, channels, use_bias=True, downsample=False, block_name='bottle_resblock'):
    layers = self.layers()
    weight_regularizer = self.weight_regularizer
    self.x = layers.BatchNormalization(name=block_name + '/batch_norm_1x1_front')(self.x)
    shortcut = layers.Activation('relu', name=block_name + '/relu_1x1_front')(self.x)
    self.x = layers.Conv2D(channels, 1, 1, 'same', kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                           use_bias=use_bias, name=block_name + '/conv_1x1_front')(shortcut)
    self.x = layers.BatchNormalization(name=block_name + '/batch_norm_3x3')(self.x)
    self.x = layers.Activation('relu', name=block_name + '/relu_3x3')(self.x)
    if downsample:
      self.x = layers.Conv2D(channels, 3, 2, 'same', kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer, use_bias=use_bias, name=block_name + '/conv_0')(self.x)
      shortcut = layers.Conv2D(channels * 4, 1, 2, 'same', kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                               use_bias=use_bias, name=block_name + '/conv_init')(shortcut)
    else:
      self.x = layers.Conv2D(channels, 3, 1, 'same', kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer, use_bias=use_bias, name=block_name + '/conv_0')(self.x)
      shortcut = layers.Conv2D(channels * 4, 1, 1, 'same', kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                               use_bias=use_bias, name=block_name + '/conv_init')(shortcut)
    self.x = layers.BatchNormalization(name=block_name + '/batch_norm_1x1_back')(self.x)
    self.x = layers.Activation('relu', name=block_name + '/relu_1x1_back')(self.x)
    self.x = layers.Conv2D(channels * 4, 1, 1, 'same', kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                           use_bias=use_bias, name=block_name + '/conv_1x1_back')(self.x)
    self.x = layers.Add()([self.x, shortcut])


class ResNetHyper(ResNet):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def model(self):
    self.res_n = 2 * self.hp.Int('depth', min_value=5, max_value=17, step=1)
    super().model()


class ResNet50(ResNet):
  def __init__(self, *args, **kwargs):
    self.res_n = 50
    super().__init__(*args, **kwargs)


class ResNet101(ResNet):
  def __init__(self, *args, **kwargs):
    self.res_n = 101
    super().__init__(*args, **kwargs)


class ResNet152(ResNet):
  def __init__(self, *args, **kwargs):
    self.res_n = 152
    super().__init__(*args, **kwargs)


class ResNet34(ResNet):
  def __init__(self, *args, **kwargs):
    self.res_n = 34
    super().__init__(*args, **kwargs)


class ResNet18(ResNet):
  def __init__(self, *args, **kwargs):
    self.res_n = 18
    super().__init__(*args, **kwargs)


class ResNet50NCHW(ResNet):
  def __init__(self, *args, **kwargs):
    global is_channel_fist
    is_channel_fist = True
    self.res_n = 50
    super().__init__(*args, **kwargs)


class ResNet101NCHW(ResNet):
  def __init__(self, *args, **kwargs):
    global is_channel_fist
    is_channel_fist = True
    self.res_n = 101
    super().__init__(*args, **kwargs)


class ResNet152NCHW(ResNet):
  def __init__(self, *args, **kwargs):
    global is_channel_fist
    is_channel_fist = True
    self.res_n = 152
    super().__init__(*args, **kwargs)


class ResNet34NCHW(ResNet):
  def __init__(self, *args, **kwargs):
    global is_channel_fist
    is_channel_fist = True
    self.res_n = 34
    super().__init__(*args, **kwargs)


class ResNet18NCHW(ResNet):
  def __init__(self, *args, **kwargs):
    global is_channel_fist
    is_channel_fist = True
    self.res_n = 18
    super().__init__(*args, **kwargs)


class ResNetCIFAR(GenericModel):
  def __init__(self, *args, **kwargs):
    super(ResNetCIFAR, self).__init__(*args, **kwargs)

  def get_residual_layer(self):
    n_to_residual = {
        20: [3],
        32: [5],
        44: [7],
        56: [9],
    }
    return n_to_residual[self.res_n] * 3

  def model(self):
    residual_list = self.get_residual_layer()
    weight_regularizer = self.weight_regularizer
    ch = 16
    self.x = self.layers().Conv2D(int(ch/self.factor), 3, 1, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                  padding="same", name='conv')(self.x)

    # block 1
    for i in range(residual_list[0]):
      self.resblock_cifar(channels=int(ch/self.factor), stride=1, downsample=False, block_name='resblock0_' + str(i))
    # block 2
    self.resblock_cifar(channels=int(ch/self.factor) * 2, stride=2, downsample=True, block_name='resblock1_0')
    for i in range(1, residual_list[1]):
      self.resblock_cifar(channels=int(ch/self.factor) * 2, stride=1, downsample=False, block_name='resblock1_' + str(i))
    # block 3
    self.resblock_cifar(channels=int(ch/self.factor) * 4, stride=2, downsample=True, block_name='resblock2_0')
    for i in range(1, residual_list[2]):
      self.resblock_cifar(channels=int(ch/self.factor) * 4, stride=1, downsample=False, block_name='resblock2_' + str(i))
    # block 4
    self.x = self.layers().BatchNormalization(name='batch_norm_last')(self.x)
    self.x = self.layers().Activation('relu', name='relu_last')(self.x)
    self.x = self.layers().GlobalAveragePooling2D()(self.x)
    self.x = self.layers().Dense(units=self.label_dim, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=True,  name='logit')(self.x)

  def resblock_cifar(self, channels, use_bias=True, stride=1, downsample=False, block_name='resblock'):
    layers = self.layers()
    weight_regularizer = self.weight_regularizer
    x_init = self.x
    self.x = layers.BatchNormalization(name=block_name + '/batch_norm_0')(self.x)
    self.x = layers.Activation('relu', name=block_name + '/relu_0')(self.x)
    if downsample:
      x_init = layers.Conv2D(channels, 3, 2, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                             use_bias=use_bias, padding='same', name=block_name + '/conv_init')(x_init)
    self.x = layers.Conv2D(channels, 3, strides=stride, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                           use_bias=use_bias, padding='same', name=block_name + '/conv_0')(self.x)
    self.x = layers.BatchNormalization(name=block_name + '/batch_norm_1')(self.x)
    self.x = layers.Activation('relu', name=block_name + '/relu_1')(self.x)
    self.x = layers.Conv2D(channels, 3, 1, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                           use_bias=use_bias, padding='same', name=block_name + '/conv_1')(self.x)
    self.x = layers.Add()([self.x, x_init])


class ResNet20CIFAR(ResNetCIFAR):
  def __init__(self, *args, **kwargs):
    self.res_n = 20
    super().__init__(*args, **kwargs)


class ResNet32CIFAR(ResNetCIFAR):
  def __init__(self, *args, **kwargs):
    self.res_n = 32
    super().__init__(*args, **kwargs)


class ResNet44CIFAR(ResNetCIFAR):
  def __init__(self, *args, **kwargs):
    self.res_n = 44
    super().__init__(*args, **kwargs)


class ResNet56CIFAR(ResNetCIFAR):
  def __init__(self, *args, **kwargs):
    self.res_n = 56
    super().__init__(*args, **kwargs)
