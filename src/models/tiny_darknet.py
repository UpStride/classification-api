import tensorflow as tf
from .generic_model import GenericModel


class TinyDarknet(GenericModel):
  def model(self):
    # First half
    self.conv2d_unit(filters=16 // self.factor, kernels=3)
    self.x = self.layers().MaxPool2D(pool_size=2, strides=2, padding='same')(self.x)
    self.conv2d_unit(filters=32 // self.factor, kernels=3)
    self.x = self.layers().MaxPool2D(pool_size=2, strides=2, padding='same')(self.x)
    self.conv2d_unit(filters=64 // self.factor, kernels=3)
    self.x = self.layers().MaxPool2D(pool_size=2, strides=2, padding='same')(self.x)
    self.conv2d_unit(filters=128 // self.factor, kernels=3)
    self.x = self.layers().MaxPool2D(pool_size=2, strides=2, padding='same')(self.x)
    self.conv2d_unit(filters=256 // self.factor, kernels=3)

    # 2nd half
    self.x = self.layers().MaxPool2D(pool_size=2, strides=2, padding='same')(self.x)
    self.conv2d_unit(filters=512 // self.factor, kernels=3)
    self.x = self.layers().MaxPool2D(pool_size=2, strides=1, padding='same')(self.x)
    self.conv2d_unit(filters=1024 // self.factor, kernels=3)

    self.x = self.layers().GlobalAveragePooling2D()(self.x)
    self.x = self.layers().Dense(units=self.label_dim, use_bias=True, name='logit')(self.x)

  def conv2d_unit(self, filters, kernels, strides=1, padding='same'):
    self.x = self.layers().Conv2D(filters, kernels, padding=padding, strides=strides, use_bias=False)(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().LeakyReLU(alpha=0.1)(self.x)
