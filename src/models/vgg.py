""" code came from https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
"""
import tensorflow as tf
from .generic_model import GenericModel


class VGG16(GenericModel):
  def model(self):
    # Block 1
    self.x = self.layers().Conv2D(64//self.factor, (3, 3), padding='same', name='block1_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Conv2D(64//self.factor, (3, 3), padding='same', name='block1_conv2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(self.x)

    # Block 2
    self.x = self.layers().Conv2D(128//self.factor, (3, 3), padding='same', name='block2_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Conv2D(128//self.factor, (3, 3), padding='same', name='block2_conv2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(self.x)

    # Block 3
    self.x = self.layers().Conv2D(256//self.factor, (3, 3), padding='same', name='block3_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Conv2D(256//self.factor, (3, 3), padding='same', name='block3_conv2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Conv2D(256//self.factor, (3, 3), padding='same', name='block3_conv3')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(self.x)

    # Block 4
    self.x = self.layers().Conv2D(512//self.factor, (3, 3), padding='same', name='block4_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Conv2D(512//self.factor, (3, 3), padding='same', name='block4_conv2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Conv2D(512//self.factor, (3, 3), padding='same', name='block4_conv3')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(self.x)

    # Block 5
    self.x = self.layers().Conv2D(512//self.factor, (3, 3), padding='same', name='block5_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Conv2D(512//self.factor, (3, 3), padding='same', name='block5_conv2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Conv2D(512//self.factor, (3, 3), padding='same', name='block5_conv3')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(self.x)

    # Classification block
    self.x = self.layers().Flatten(name='flatten')(self.x)
    self.x = self.layers().Dense(4096//self.factor, name='fc1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Dense(4096//self.factor, name='fc2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Dense(self.label_dim, name='predictions')(self.x)
