import tensorflow as tf
# from kerastuner.applications import HyperResNet
from .generic_model import GenericModel


class SimpleHyper(GenericModel):
  def model(self):
    self.x = self.layers().Conv2D(self.hp.Int('conv1_filter',
                                              min_value=32//self.factor,
                                              max_value=512//self.factor,
                                              step=32//self.factor), (5, 5), 2, padding='same',
                                  use_bias=False,
                                  name='conv_1')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    for i in range(self.hp.Int('repeat_conv',
                               min_value=1,
                               max_value=3,
                               step=1)):
      self.x = self.layers().Conv2D(self.hp.Int('conv_filter',
                                                min_value=32//self.factor,
                                                max_value=512//self.factor,
                                                step=32//self.factor), (3, 3), padding='same',
                                    use_bias=False)(self.x)
      self.x = self.layers().BatchNormalization()(self.x)
      self.x = self.layers().Activation('relu')(self.x)
      self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    self.x = self.layers().Flatten()(self.x)
    self.x = self.layers().Dense(self.label_dim,
                                 use_bias=True,
                                 name='dense_1')(self.x)


class ResNetV2Hyper(GenericModel):
  """code from https://github.com/keras-team/keras-tuner/blob/master/kerastuner/applications/resnet.py
  """

  def model(self):
    conv3_depth = self.hp.Choice('conv3_depth', [4, 8])
    conv4_depth = self.hp.Choice('conv4_depth', [6, 23, 36])
    factor = self.hp.Int('factor', min_value=1, max_value=8, step=1)
    preact = True
    use_bias = True

    # Model definition.
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    # Initial conv2d block.
    self.x = self.layers().ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(self.x)
    self.x = self.layers().Conv2D(64 // factor, 7, strides=2, use_bias=use_bias, name='conv1_conv')(self.x)
    self.x = self.layers().ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(self.x)
    self.x = self.layers().MaxPooling2D(3, strides=2, name='pool1_pool')(self.x)

    # Middle hypertunable stack.
    self.x = stack2(self.layers(), self.x, 64 // factor, 3, name='conv2')
    self.x = stack2(self.layers(), self.x, 128 // factor, conv3_depth, name='conv3')
    self.x = stack2(self.layers(), self.x, 256 // factor, conv4_depth, name='conv4')
    self.x = stack2(self.layers(), self.x, 512 // factor, 3, stride1=1, name='conv5')

    # Top of the model.
    self.x = self.layers().BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='post_bn')(self.x)
    self.x = self.layers().Activation('relu', name='post_relu')(self.x)

    pooling = self.hp.Choice('pooling', ['avg', 'max'], default='avg')
    if pooling == 'avg':
      self.x = self.layers().GlobalAveragePooling2D(name='avg_pool')(self.x)
    elif pooling == 'max':
      self.x = self.layers().GlobalMaxPooling2D(name='max_pool')(self.x)

    self.x = self.layers().Dense(self.label_dim, activation='softmax', name='probs')(self.x)


def block2(layers, x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
  """A residual block.
  # Arguments
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default False, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
  # Returns
      Output tensor for the residual block.
  """
  bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

  preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
  preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

  if conv_shortcut is True:
    shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
  else:
    shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

  x = layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
  x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
  x = layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False, name=name + '_2_conv')(x)
  x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Activation('relu', name=name + '_2_relu')(x)

  x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
  x = layers.Add(name=name + '_out')([shortcut, x])
  return x


def stack2(layers, x, filters, blocks, stride1=2, name=None):
  """A set of stacked residual blocks.
  # Arguments
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
  # Returns
      Output tensor for the stacked blocks.
  """
  x = block2(layers, x, filters, conv_shortcut=True, name=name + '_block1')
  for i in range(2, blocks):
    x = block2(layers, x, filters, name=name + '_block' + str(i))
  x = block2(layers, x, filters, stride=stride1, name=name + '_block' + str(blocks))
  return x
