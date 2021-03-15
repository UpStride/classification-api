""" code came from https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
"""
import tensorflow as tf
from .generic_model import GenericModelBuilder


class VGG16(GenericModelBuilder):
  def model(self, x):
    # Block 1
    x = self.layers.Conv2D(64//self.factor, (3, 3), padding='same', name='block1_conv1')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Conv2D(64//self.factor, (3, 3), padding='same', name='block1_conv2')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = self.layers.Conv2D(128//self.factor, (3, 3), padding='same', name='block2_conv1')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Conv2D(128//self.factor, (3, 3), padding='same', name='block2_conv2')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = self.layers.Conv2D(256//self.factor, (3, 3), padding='same', name='block3_conv1')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Conv2D(256//self.factor, (3, 3), padding='same', name='block3_conv2')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Conv2D(256//self.factor, (3, 3), padding='same', name='block3_conv3')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = self.layers.Conv2D(512//self.factor, (3, 3), padding='same', name='block4_conv1')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Conv2D(512//self.factor, (3, 3), padding='same', name='block4_conv2')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Conv2D(512//self.factor, (3, 3), padding='same', name='block4_conv3')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = self.layers.Conv2D(512//self.factor, (3, 3), padding='same', name='block5_conv1')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Conv2D(512//self.factor, (3, 3), padding='same', name='block5_conv2')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Conv2D(512//self.factor, (3, 3), padding='same', name='block5_conv3')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = self.layers.Flatten(name='flatten')(x)
    x = self.layers.Dense(4096//self.factor, name='fc1')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Dense(4096//self.factor, name='fc2')(x)
    x = self.layers.Activation('relu')(x)
    return x
