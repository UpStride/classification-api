import unittest
import tensorflow as tf
import numpy as np
from .pdart import DropPath

class TestDropPath(unittest.TestCase):
  def test(self):
    x = tf.ones(shape=(10000, 1, 1, 1))
    drop_path_prob = tf.convert_to_tensor(0.3)
    y = DropPath()([x, drop_path_prob])
    # Mean of y shouldn't change much
    self.assertAlmostEqual(tf.reduce_sum(y).numpy()/10000, 1., 1)

  def test_nn(self):
    """ Create a single layer NN
    """
    # define NN
    x = tf.keras.layers.Input(shape=(1, 1, 1))
    drop_path_prob = tf.keras.layers.Input(shape=[])
    y = DropPath()([x, drop_path_prob])
    model = tf.keras.Model(inputs=[x, drop_path_prob], outputs=y)

    # run NN
    inputs = [tf.ones(shape=(1000, 1, 1, 1)), tf.convert_to_tensor(0.3)]
    outputs = model(inputs)
    outputs_mean = tf.reduce_mean(outputs)
    self.assertAlmostEqual(outputs_mean.numpy(), 1., 1)

    inputs = [tf.ones(shape=(1000, 1, 1, 1)), tf.convert_to_tensor(0.)]
    outputs2 = model(inputs)
    np.array_equal(np.ones(shape=(1000, 1, 1, 1)), outputs2.numpy())
