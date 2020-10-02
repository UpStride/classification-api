import tensorflow as tf
from .generic_model import GenericModel


class AlexNet(GenericModel):
  def model(self):
    # note regarding batch norm : in the official implementation, there are 2 batchnorms.
    # However, it seems they are hurting the training when using with upstride, so for now there are commented.
    # Maybe it will change some day, it's why they are commented and not removed
    self.x = self.layers().Conv2D(96//self.factor, (11, 11), 4, padding='same',
                                  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_1')(self.x)
    #x = tf.keras.layers.BatchNormalization()(x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 2 - Conv
    self.x = self.layers().Conv2D(256//self.factor, (5, 5), padding='same',
                                  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_2')(self.x)
    #x = tf.keras.layers.BatchNormalization()(x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 3 - Conv
    self.x = self.layers().Conv2D(384//self.factor, (3, 3), padding='same',
                                  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_3')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    # Layer 4 - Conv
    self.x = self.layers().Conv2D(384//self.factor, (3, 3), padding='same',
                                  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_4')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    # Layer 5 - Conv
    self.x = self.layers().Conv2D(256//self.factor, (3, 3), padding='same',
                                  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_5')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 6 - Fully connected
    self.x = self.layers().Flatten()(self.x)
    self.x = self.layers().Dense(4096//self.factor,
                                 kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=False,
                                 name='dense_1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Dropout(0.5, seed=42)(self.x)
    # Layer 7 - Fully connected
    self.x = self.layers().Dense(4096//self.factor,
                                 kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=False,
                                 name='dense_2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Dropout(0.5, seed=42)(self.x)
    self.x = self.layers().Dense(self.label_dim,
                                 kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=True,
                                 name='dense_3')(self.x)


class AlexNetQ(GenericModel):
  def model(self):
    self.x = self.layers().Conv2D(96//self.factor, (11, 11), 4, padding='same',
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_1')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 2 - Conv
    self.x = self.layers().Conv2D(256//self.factor, (5, 5), padding='same',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_2')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 3 - Conv
    self.x = self.layers().Conv2D(384//self.factor, (3, 3), padding='same',
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_3')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    # Layer 4 - Conv
    self.x = self.layers().Conv2D(384//self.factor, (3, 3), padding='same',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_4')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    # Layer 5 - Conv
    self.x = self.layers().Conv2D(256//self.factor, (3, 3), padding='same',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_5')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 6 - Fully connected
    self.x = self.layers().Flatten()(self.x)
    self.x = self.layers().Dense(4096//self.factor,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=False,
                                 name='dense_1')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Dropout(0.5, seed=42)(self.x)
    # Layer 7 - Fully connected
    self.x = self.layers().Dense(4096//self.factor,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=False,
                                 name='dense_2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Dropout(0.5, seed=42)(self.x)
    self.x = self.layers().Dense(self.label_dim,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=True,
                                 name='dense_3')(self.x)

class AlexNetNCHW(GenericModel):
  def model(self):
    self.x = tf.transpose(self.x, [0, 3, 1, 2])
    tf.keras.backend.set_image_data_format('channels_first')

    self.x = self.layers().Conv2D(96//self.factor, (11, 11), 4, padding='same',
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_1')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 2 - Conv
    self.x = self.layers().Conv2D(256//self.factor, (5, 5), padding='same',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_2')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 3 - Conv
    self.x = self.layers().Conv2D(384//self.factor, (3, 3), padding='same',
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_3')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    # Layer 4 - Conv
    self.x = self.layers().Conv2D(384//self.factor, (3, 3), padding='same',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_4')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    # Layer 5 - Conv
    self.x = self.layers().Conv2D(256//self.factor, (3, 3), padding='same',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_5')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 6 - Fully connected
    self.x = self.layers().Flatten()(self.x)
    self.x = self.layers().Dense(4096//self.factor,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=False,
                                 name='dense_1')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Dropout(0.5, seed=42)(self.x)
    # Layer 7 - Fully connected
    self.x = self.layers().Dense(4096//self.factor,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=False,
                                 name='dense_2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Dropout(0.5, seed=42)(self.x)
    self.x = self.layers().Dense(self.label_dim,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=True,
                                 name='dense_3')(self.x)

class AlexNetToy(GenericModel):
  def model(self):
    # This model is a mini version of the AlexNet
    self.x = self.layers().Conv2D(96//self.factor, (11, 11), 4, padding='valid',
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_1')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 2 - Conv
    self.x = self.layers().Conv2D(128//self.factor, (5, 5), padding='valid',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_2')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 3 - Conv
    self.x = self.layers().Conv2D(192//self.factor, (3, 3), padding='valid',
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_3')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    # Layer 4 - Conv
    self.x = self.layers().Conv2D(128//self.factor, (3, 3), padding='valid',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_5')(self.x)
    self.x = self.layers().BatchNormalization()(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().MaxPooling2D((3, 3), strides=(2, 2))(self.x)
    # Layer 5 - Fully connected
    self.x = self.layers().Flatten()(self.x)
    self.x = self.layers().Dense(2048//self.factor,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=False,
                                 name='dense_2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    self.x = self.layers().Dropout(0.5, seed=42)(self.x)
    self.x = self.layers().Dense(self.label_dim,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=True,
                                 name='dense_3')(self.x)
