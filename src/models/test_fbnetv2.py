import unittest
import tensorflow as tf
import numpy as np
from . import fbnetv2
import scipy.stats as ss


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
    
class TestSplitTtrainableWeights(unittest.TestCase):
  def test_split_trainable_weights(self):
    layer0 = tf.keras.layers.Input((32, 32, 3))
    layer1 = tf.keras.layers.Conv2D(8, kernel_size=3, strides=1, padding='same')
    layer2 = fbnetv2.ChannelMasking(2, 8, 2, 'abc', gumble_noise=False)
    model = tf.keras.Sequential([layer0, layer1, layer2])

    weights, arch_params = fbnetv2.split_trainable_weights(model)

    true_total_weight_param = 3*3*3*8+8
    true_total_arch_param = len(range(2, 8+1, 2))

    # calculate number of weight params returned by  the function
    total_weight_params = 0
    for w in weights:
      total_weight_params += np.prod(w.shape.as_list())

    # calculate number of architecture params returned by  the function
    total_arch_params = 0
    for p in arch_params:
      total_arch_params += np.prod(p.shape.as_list())

    self.assertEqual(total_arch_params, true_total_arch_param)
    self.assertEqual(total_weight_params, true_total_weight_param)
    self.assertEqual(total_weight_params+total_arch_params, true_total_weight_param+true_total_arch_param)

  def test_not_arch_params(self):
    layer0 = tf.keras.layers.Input((32, 32, 3))
    layer1 = tf.keras.layers.Conv2D(8, kernel_size=3, strides=1, padding='same')
    model = tf.keras.Sequential([layer0, layer1])

    # check if it raises error when there is no architectural parameters by the name 'alpha'
    self.assertRaises(ValueError, fbnetv2.split_trainable_weights, model, arch_params_name='alpha')

class TestGumbelSoftmax(unittest.TestCase):
  def testSampling(self):
    fbnetv2.define_temperature(5.)
    noise = 0.0001
    logits = tf.constant([-1., 0.5, 1.])

    g = fbnetv2.gumbel_softmax(logits, gumble_noise=False)

    self.assertEqual(logits.shape.as_list(), g.shape.as_list())
    self.assertEqual(g.numpy().mean(), 1.0)
    self.assertSequenceEqual(g.numpy().tolist(), tf.math.softmax((logits+noise)/5.).numpy().tolist())

  def testUniformLikeDist(self):
    # set temperature values to high to see Uniform like distribution
    fbnetv2.define_temperature(5.0)
    logits = tf.constant([-2., 2., -2.5, -2.])

    g = fbnetv2.gumbel_softmax(logits, gumble_noise=False)

    # Test the distribution using Kolmogorove-Smirnov test as 
    # described here (https://stackoverflow.com/questions/22392562/how-can-check-the-distribution-of-a-variable-in-python)

    _, p = ss.kstest(g.numpy(), ss.uniform.cdf)

    th = 0.05

    # Check the significance, if the signifncace are higher than threshold(th), we can assume it to be uniform
    self.assertGreater(p, th)

  def testOnehotLikeDist(self):
    # set temperature values to high to see Onehot like distribution
    fbnetv2.define_temperature(0.00001)
    logits = tf.constant([-2., 2., -2.5, -2.])

    g = fbnetv2.gumbel_softmax(logits, gumble_noise=False)

    _, p = ss.kstest(g.numpy(), ss.uniform.cdf)

    th = 0.05

    # Check the significance, if the signifncace are lesser than threshold(th), 
    # we can assume it to be not uniform thus onhot
    self.assertLess(p, th)
  