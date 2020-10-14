import unittest
import tensorflow as tf
import numpy as np
from . import fbnetv2


class TestBinaryVector(unittest.TestCase):
  def test_create_binary_vector(self):
    binary_vectors = fbnetv2.create_binary_vector(channel_sizes=[2, 7, 10], dtype=tf.float32)
    # check that we have 3 vectors
    self.assertEqual(len(binary_vectors), 3)
    # check the types of vectors
    for i in range(3):
      self.assertEqual(binary_vectors[i].dtype, tf.float32)
    # check the vector content
    self.assertTrue(np.array_equal(binary_vectors[0].numpy(), [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))
    self.assertTrue(np.array_equal(binary_vectors[1].numpy(), [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.]))
    self.assertTrue(np.array_equal(binary_vectors[2].numpy(), [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))


class TestGetMask(unittest.TestCase):
  def test_get_mask(self):
    binary_vectors = fbnetv2.create_binary_vector(channel_sizes=[1, 2, 4], dtype=tf.float32)
    g = [2., 3., 5.]
    mask = fbnetv2.get_mask(binary_vectors, g)
    self.assertEqual(mask.dtype, tf.float32)
    self.assertTrue(np.array_equal(mask, [10.,  8.,  5.,  5.]))

class TestChannelMasking(unittest.TestCase):
  def test_init(self):
    cm = fbnetv2.ChannelMasking(1, 5, 2, 'toto')
    self.assertEqual(cm.channel_sizes, [1, 3, 5])
    
  def test_build_manual(self):
    cm = fbnetv2.ChannelMasking(1, 5, 2, 'toto')
    cm.build((15, 15, 3))
    self.assertTrue(np.array_equal(cm.alpha.numpy(), [1., 1., 1.]))
    self.assertTrue(np.array_equal(cm.binary_vectors[0].numpy(), [1., 0., 0., 0., 0.]))
    self.assertTrue(np.array_equal(cm.binary_vectors[1].numpy(), [1., 1., 1., 0., 0.]))
    self.assertTrue(np.array_equal(cm.binary_vectors[2].numpy(), [1., 1., 1., 1., 1.]))

  def test_build_keras(self):
    cm = fbnetv2.ChannelMasking(1, 5, 2, 'toto', gumble_noise=False)
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(5, (3, 3), padding='same', use_bias=False), cm])
    model(tf.zeros((1, 24, 24, 3), dtype=tf.float32)) # build is called here
    self.assertTrue(np.array_equal(cm.alpha.numpy(), [1., 1., 1.]))
    self.assertTrue(np.array_equal(cm.binary_vectors[0].numpy(), [1., 0., 0., 0., 0.]))
    self.assertTrue(np.array_equal(cm.binary_vectors[1].numpy(), [1., 1., 1., 0., 0.]))
    self.assertTrue(np.array_equal(cm.binary_vectors[2].numpy(), [1., 1., 1., 1., 1.]))

  def test_call(self):
    cm = fbnetv2.ChannelMasking(1, 5, 2, 'toto', gumble_noise=False)
    model = tf.keras.Sequential([cm])
    out = model(tf.ones((1, 3, 3, 5), dtype=tf.float32)) # build is called here
    # check g parameter
    for e in cm.g.numpy():
      self.assertAlmostEqual(e, 1/3)
    # check output of the model
    self.assertEqual(out.shape, (1, 3, 3, 5))
    components = out[0, 0, 0]
    self.assertAlmostEqual(components.numpy()[0], 1)
    self.assertAlmostEqual(components.numpy()[1], 2/3)
    self.assertAlmostEqual(components.numpy()[2], 2/3)
    self.assertAlmostEqual(components.numpy()[3], 1/3)
    self.assertAlmostEqual(components.numpy()[4], 1/3)
    