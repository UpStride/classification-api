import tensorflow as tf
from .generic_model import GenericModelBuilder


class TinyDarknet(GenericModelBuilder):
  def model(self, x):
    # First half
    self.conv2d_unit(filters=16 // self.factor, kernels=3)
    x = self.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    self.conv2d_unit(filters=32 // self.factor, kernels=3)
    x = self.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    self.conv2d_unit(filters=64 // self.factor, kernels=3)
    x = self.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    self.conv2d_unit(filters=128 // self.factor, kernels=3)
    x = self.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    self.conv2d_unit(filters=256 // self.factor, kernels=3)

    # 2nd half
    x = self.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    self.conv2d_unit(filters=512 // self.factor, kernels=3)
    x = self.layers.MaxPool2D(pool_size=2, strides=1, padding='same')(x)
    self.conv2d_unit(filters=1024 // self.factor, kernels=3)

    x = self.layers.GlobalAveragePooling2D()(x)
    return x

  def conv2d_unit(self, x, filters, kernels, strides=1, padding='same'):
    x = self.layers.Conv2D(filters, kernels, padding=padding, strides=strides, use_bias=False)(x)
    x = self.layers.BatchNormalization()(x)
    x = self.layers.LeakyReLU(alpha=0.1)(x)
    return x
