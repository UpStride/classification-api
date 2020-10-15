import unittest
import tensorflow as tf
from src.models.fbnetv2 import ChannelMasking
from src.losses import flops_loss


class TestLosses(unittest.TestCase):
  def test_flops_loss(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(24, 24, 3)),
        tf.keras.layers.Conv2D(3, (3, 3), padding='same', use_bias=False),
        ChannelMasking(1, 3, 1, "hello", gumble_noise=False)
    ])

    model(tf.zeros((1, 24, 24, 3), dtype=tf.float32))
    l = flops_loss(model)
    conv_flops = 3*3*3*3*24*24 * 2
    self.assertLess(l, conv_flops)
    self.assertAlmostEqual(float(l), conv_flops * ((1/3)**2 + (1/3)*(2/3)+(1/3)))

    model.layers[1].g = tf.convert_to_tensor([1., 0., 0.], dtype=tf.float32)
    l = flops_loss(model)
    self.assertAlmostEqual(float(l), conv_flops * ((1/3)))

    model.layers[1].g = tf.convert_to_tensor([0., 1., 0.], dtype=tf.float32)
    l = flops_loss(model)
    self.assertAlmostEqual(float(l), conv_flops * ((2/3)))

    model.layers[1].g = tf.convert_to_tensor([0., 0., 1.], dtype=tf.float32)
    l = flops_loss(model)
    self.assertAlmostEqual(float(l), conv_flops)

  def test_flops_with_intermediate_ops_loss(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(24, 24, 3)),
        tf.keras.layers.Conv2D(3, (3, 3), padding='same', use_bias=False),
        tf.keras.layers.ReLU(),
        ChannelMasking(1, 3, 1, "hello", gumble_noise=False)
    ])
    model(tf.zeros((1, 24, 24, 3), dtype=tf.float32))
    l = flops_loss(model)
    conv_flops = 3*3*3*3*24*24 * 2
    self.assertLess(l, conv_flops)
    self.assertAlmostEqual(float(l), conv_flops * ((1/3)**2 + (1/3)*(2/3)+(1/3)))
