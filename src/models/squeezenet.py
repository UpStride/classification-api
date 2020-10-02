import tensorflow as tf
from .generic_model import GenericModel


class SqueezeNet(GenericModel):
  def model(self):
    self.x = self.layers().Conv2D(filters=64 // self.factor, kernel_size=3, strides=2, padding='valid', name='conv1')(self.x)

    self.x = self.layers().Activation('relu', name='relu_conv1')(self.x)
    self.x = self.layers().MaxPooling2D(pool_size=3, strides=2, name='pool1')(self.x)

    self.fire_module(fire_id=2, s1x1=16 // self.factor, e1x1=64 // self.factor, e3x3=64 // self.factor)
    self.fire_module(fire_id=3, s1x1=16 // self.factor, e1x1=64 // self.factor, e3x3=64 // self.factor)
    self.x = self.layers().MaxPooling2D(pool_size=3, strides=2, name='pool3')(self.x)

    self.fire_module(fire_id=4, s1x1=32 // self.factor, e1x1=128 // self.factor, e3x3=128 // self.factor)
    self.fire_module(fire_id=5, s1x1=32 // self.factor, e1x1=128 // self.factor, e3x3=128 // self.factor)
    self.x = self.layers().MaxPooling2D(pool_size=3, strides=2, name='pool5')(self.x)

    self.fire_module(fire_id=6, s1x1=48 // self.factor, e1x1=192 // self.factor, e3x3=192 // self.factor)
    self.fire_module(fire_id=7, s1x1=48 // self.factor, e1x1=192 // self.factor, e3x3=192 // self.factor)
    self.fire_module(fire_id=8, s1x1=64 // self.factor, e1x1=256 // self.factor, e3x3=256 // self.factor)
    self.fire_module(fire_id=9, s1x1=64 // self.factor, e1x1=256 // self.factor, e3x3=256 // self.factor)

    self.x = self.layers().Dropout(0.5, name='drop9')(self.x)
    self.x = self.layers().Conv2D(filters=self.label_dim, kernel_size=1, padding='valid', name='conv10')(self.x)
    self.x = self.layers().Activation('relu', name='relu_conv10')(self.x)
    self.x = self.layers().GlobalAveragePooling2D()(self.x)

  def fire_module(self, fire_id, s1x1=16, e1x1=64, e3x3=64):
    """tf.keras

    Args:
        fire_id: id of fire module
        s1x1: filter size of squeeze layer
        e1x1: filter size of 1x1 expand layer
        e3x3: filter size of 3x3 expand layer
    Returns:
        a keras tensor
    """

    s_id = 'fire' + str(fire_id) + '/'

    self.x = self.layers().Conv2D(filters=s1x1, kernel_size=1, padding='valid', name=s_id + 'squeeze1x1_conv')(self.x)
    self.x = self.layers().Activation('relu', name=s_id + 'squeeze1x1_relu')(self.x)

    expand1x1 = self.layers().Conv2D(filters=e1x1, kernel_size=1, padding='valid', name=s_id + 'expand1x1_conv')(self.x)
    expand1x1 = self.layers().Activation('relu', name=s_id + 'expand1x1_relu')(expand1x1)

    expand3x3 = self.layers().Conv2D(filters=e3x3, kernel_size=3, padding='same', name=s_id + 'expand3x3_conv')(self.x)
    expand3x3 = self.layers().Activation('relu', name=s_id + 'expand3x3_relu')(expand3x3)

    self.x = self.layers().Concatenate(axis=3, name=s_id + 'concat')([expand1x1, expand3x3])
