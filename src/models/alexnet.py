import tensorflow as tf
from .generic_model import GenericModelBuilder


class AlexNet(GenericModelBuilder):
  def model(self, x):
    # note regarding batch norm : in the official implementation, there are 2 batchnorms.
    # However, it seems they are hurting the training when using with upstride, so for now there are commented.
    # Maybe it will change some day, it's why they are commented and not removed
    x = self.layers.Conv2D(96//self.factor, (11, 11), 4, padding='same',
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                           bias_initializer=tf.keras.initializers.zeros(),
                           use_bias=False,
                           name='conv_1')(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Layer 2 - Conv
    x = self.layers.Conv2D(256//self.factor, (5, 5), padding='same',
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                           bias_initializer=tf.keras.initializers.ones(),
                           use_bias=False,
                           name='conv_2')(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Layer 3 - Conv
    x = self.layers.Conv2D(384//self.factor, (3, 3), padding='same',
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                           bias_initializer=tf.keras.initializers.zeros(),
                           use_bias=False,
                           name='conv_3')(x)
    x = self.layers.Activation('relu')(x)
    # Layer 4 - Conv
    x = self.layers.Conv2D(384//self.factor, (3, 3), padding='same',
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                           bias_initializer=tf.keras.initializers.ones(),
                           use_bias=False,
                           name='conv_4')(x)
    x = self.layers.Activation('relu')(x)
    # Layer 5 - Conv
    x = self.layers.Conv2D(256//self.factor, (3, 3), padding='same',
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                           bias_initializer=tf.keras.initializers.ones(),
                           use_bias=False,
                           name='conv_5')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Layer 6 - Fully connected
    x = self.layers.Flatten()(x)
    x = self.layers.Dense(4096//self.factor,
                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                          bias_initializer=tf.keras.initializers.ones(),
                          use_bias=False,
                          name='dense_1')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Dropout(0.5, seed=42)(x)
    # Layer 7 - Fully connected
    x = self.layers.Dense(4096//self.factor,
                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42),
                          bias_initializer=tf.keras.initializers.ones(),
                          use_bias=False,
                          name='dense_2')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Dropout(0.5, seed=42)(x)
    return x


class AlexNetQ(GenericModelBuilder):
  def model(self, x):
    x = self.layers.Conv2D(96//self.factor, (11, 11), 4, padding='same',
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_1')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis)(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Layer 2 - Conv
    x = self.layers.Conv2D(256//self.factor, (5, 5), padding='same',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_2')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis)(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Layer 3 - Conv
    x = self.layers.Conv2D(384//self.factor, (3, 3), padding='same',
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_3')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis)(x)
    x = self.layers.Activation('relu')(x)
    # Layer 4 - Conv
    x = self.layers.Conv2D(384//self.factor, (3, 3), padding='same',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_4')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis)(x)
    x = self.layers.Activation('relu')(x)
    # Layer 5 - Conv
    x = self.layers.Conv2D(256//self.factor, (3, 3), padding='same',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_5')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis)(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Layer 6 - Fully connected
    x = self.layers.Flatten()(x)
    x = self.layers.Dense(4096//self.factor,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=False,
                                 name='dense_1')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis)(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Dropout(0.5, seed=42)(x)
    # Layer 7 - Fully connected
    x = self.layers.Dense(4096//self.factor,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=False,
                                 name='dense_2')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Dropout(0.5, seed=42)(x)
    return x

class AlexNetToy(GenericModelBuilder):
  def model(self, x):
    # This model is a mini version of the AlexNet
    x = self.layers.Conv2D(96//self.factor, (11, 11), 4, padding='valid',
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_1')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis)(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Layer 2 - Conv
    x = self.layers.Conv2D(128//self.factor, (5, 5), padding='valid',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_2')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis)(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Layer 3 - Conv
    x = self.layers.Conv2D(192//self.factor, (3, 3), padding='valid',
                                  bias_initializer=tf.keras.initializers.zeros(),
                                  use_bias=False,
                                  name='conv_3')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis)(x)
    x = self.layers.Activation('relu')(x)
    # Layer 4 - Conv
    x = self.layers.Conv2D(128//self.factor, (3, 3), padding='valid',
                                  bias_initializer=tf.keras.initializers.ones(),
                                  use_bias=False,
                                  name='conv_5')(x)
    x = self.layers.BatchNormalization(axis=self.channel_axis)(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Layer 5 - Fully connected
    x = self.layers.Flatten()(x)
    x = self.layers.Dense(2048//self.factor,
                                 bias_initializer=tf.keras.initializers.ones(),
                                 use_bias=False,
                                 name='dense_2')(x)
    x = self.layers.Activation('relu')(x)
    x = self.layers.Dropout(0.5, seed=42)(x)
    return x
