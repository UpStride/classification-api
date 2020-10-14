import unittest
import shutil 
import tempfile
import yaml
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
    
class TestExponentialDecay(unittest.TestCase):
  def non_increasing(self, decay):
    """Checks if all the function provided are non increasing over a range

    Args:
        decay (instance): Instance of the decay to be tested

    Returns:
        bool : Compares ith and i + 1st element of the value_list
        returns True if all i >= i+1 else False
    """
    value_list = [decay(i) for i in range(1,100)]
    return all([i >= j for i, j in zip(value_list, value_list[1:])])

  def test_exponential_decay(self):
    decay = fbnetv2.exponential_decay(5, 1, 0.956)

    self.assertEqual(decay(0), 5)  
    self.assertAlmostEqual(decay(10), 3.188,places=3)

    # test function is not increasing over number of epochs
    self.assertTrue(self.non_increasing(decay), True)

    # Negative test to ensure decay rate greater than 1 is increasing
    decay = fbnetv2.exponential_decay(5, 1, 1.1)
    self.assertFalse(self.non_increasing(decay), False)

class TestPostTrainingAnalysis(unittest.TestCase):
  def test_post_training_anaysis(self):
    cm1 = fbnetv2.ChannelMasking(1, 5, 2, 'toto_1_savable', gumble_noise=False)
    cm2 = fbnetv2.ChannelMasking(8, 16, 4, 'toto_2_savable', gumble_noise=False)
    model = tf.keras.Sequential(
      [tf.keras.layers.Conv2D(5, (3, 3), padding='same', use_bias=False), 
      cm1,
      tf.keras.layers.Conv2D(16, (3, 3), padding='same', use_bias=False), 
      cm2,
      ])
    model(tf.zeros((1, 24, 24, 3), dtype=tf.float32)) # build is called here
    tmpdir = tempfile.mkdtemp() 
    tmpfile = tmpdir + "/test.yaml"
    fbnetv2.post_training_analysis(model,tmpfile)
    with open(tmpfile, 'r') as f:
      read = yaml.safe_load(f)
    self.assertDictEqual({"toto_1": 1, "toto_2": 8}, read)
    shutil.rmtree(tmpdir)