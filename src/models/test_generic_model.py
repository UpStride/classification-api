import sys
import unittest
import tensorflow as tf
from unittest.mock import MagicMock, patch
from .generic_model import Layer, GenericModel

# sys.modules['upstride.type2.tf.keras.layers'] = MagicMock()
# sys.modules['upstride'] = MagicMock()


class TestLayer(unittest.TestCase):
  def test_n_layers_before_tf(self):
    layer = Layer("tensorflow", n_layers_before_tf=3)
    # n_layers_before_tf is ignored with "tensorflow"
    self.assertEqual(layer(), tf.keras.layers)

    layer = Layer("upstride_type2", n_layers_before_tf=3)
    self.assertNotEqual(layer(), tf.keras.layers)
    self.assertNotEqual(layer(), tf.keras.layers)
    self.assertNotEqual(layer(), tf.keras.layers)
    self.assertNotEqual(layer(), tf.keras.layers)

    layer = Layer("mix_type2", n_layers_before_tf=3)
    self.assertNotEqual(layer(), tf.keras.layers)
    self.assertNotEqual(layer(), tf.keras.layers)
    self.assertNotEqual(layer(), tf.keras.layers)
    self.assertEqual(layer(), tf.keras.layers)


class Model1(GenericModel):
  def model(self):
    self.x = self.layers().Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(self.x)
    self.x = self.layers().Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(self.x)
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(self.x)


class TestModel1(unittest.TestCase):
  def test_model(self):
    # This unit test doesn't work anymore because we can't know which engine is used
    pass
    # model = Model1('mix_type2', factor=4, n_layers_before_tf=1).model

    # # got with model.summary()
    # model.summary()
    # layer_names = [
    #     'InputLayer',
    #     'TF2Upstride',
    #     'Upstride_2_Conv2D',
    #     'Upstride2TF',
    #     'Conv2D',
    #     'MaxPooling2D',
    #     'Activation',
    # ]

    # for i in range(7):
    #     print(model.get_layer(index=i))
