import unittest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors import InvalidArgumentError
from . import augmentations


class TestAugmentations(unittest.TestCase):
  @classmethod
  def setUp(self):
    self.mean_rgb = [0.485, 0.456, 0.406]
    self.stddev_rgb = [0.229, 0.224, 0.225]
    self.image = np.ones((224, 224, 3), dtype=np.uint8) * 255.
    self.rand_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.int)
    self.INTERPOLATION_METHODS = [
        'bilinear',
        'lanczos3',
        'lanczos5',
        'lanczos5',
        'bicubic',
        'gaussian',
        'nearest',
        'area',
        'mitchellcubic'
    ]

  def test_normalize(self):
    red_value = (1 - self.mean_rgb[0]) / self.stddev_rgb[0]
    green_value = (1 - self.mean_rgb[1]) / self.stddev_rgb[1]
    blue_value = (1 - self.mean_rgb[2]) / self.stddev_rgb[2]
    # case 1 : z score normalization
    config = {
        'mean': self.mean_rgb,
        'std': self.stddev_rgb,
        'scale_in_zero_to_one': False,
        'only_subtract_mean': False
    }
    output = augmentations.Normalize(config)(self.image)
    self.assertAlmostEqual(output[100, 50, 0], red_value, places=5)
    self.assertAlmostEqual(output[100, 50, 1], green_value, places=5)
    self.assertAlmostEqual(output[100, 50, 2], blue_value, places=5)

  def test_normalize_scale(self):
    mean_rgb = [0., 0., 0.]
    stddev_rgb = [1., 1., 1.]
    # case 2 : with scale between 0 and 1
    config = {
        'mean': mean_rgb,
        'std': stddev_rgb,
        'scale_in_zero_to_one': True,
        'only_subtract_mean': False
    }
    output = augmentations.Normalize(config)(self.image)
    self.assertAlmostEqual(output[100, 50, 0], 1.)
    self.assertAlmostEqual(output[100, 50, 1], 1.)
    self.assertAlmostEqual(output[100, 50, 2], 1.)

  def test_normalize_center(self):
    red_value = (1. - self.mean_rgb[0])
    green_value = (1. - self.mean_rgb[1])
    blue_value = (1. - self.mean_rgb[2])
    # case 3 : normalize and center the data
    config = {
        'mean': self.mean_rgb,
        'std': self.stddev_rgb,
        'scale_in_zero_to_one': False,
        'only_subtract_mean': True
    }
    output = augmentations.Normalize(config)(self.image)
    self.assertAlmostEqual(output[100, 50, 0], red_value, places=5)
    self.assertAlmostEqual(output[100, 50, 1], green_value, places=5)
    self.assertAlmostEqual(output[100, 50, 2], blue_value, places=5)

  def test_resize(self):
    size = [100, 50]
    # case 1
    for i in self.INTERPOLATION_METHODS:
      config = {
          "size": size,
          'interpolation': i
      }
      output = augmentations.Resize(config)(self.image)
      self.assertEqual(output.shape, (100, 50, 3))
    # case 2 interpolation not found in the list
    with self.assertRaises(ValueError):
      config = {
          "size": size,
          'interpolation': 'test'
      }
      output = augmentations.Resize(config)(self.image)

  def test_resize_then_random_crop(self):
    config = {
        'size': [200, 200],
        'crop_size': [100, 50, 3],
        'interpolation': None
    }

    # case 1
    for i in self.INTERPOLATION_METHODS:
      config['interpolation'] = i
      output = augmentations.ResizeThenRandomCrop(config)(self.image)
      self.assertEqual(output.shape, [100, 50, 3])

    # case 2 interpolation not found in the list
    with self.assertRaises(ValueError):
      config['interpolation'] = 'test'
      output = augmentations.ResizeThenRandomCrop(config)(self.image)

    # case 3 fail when random crop is greater than the resized crop size
    crop_size = [300, 300, 3]
    with self.assertRaises(InvalidArgumentError):
      config['interpolation'] = 'bilinear'
      config['crop_size'] = crop_size
      output = augmentations.ResizeThenRandomCrop(config)(self.image)

  def test_randomcrop_then_resize(self):
    size = [200, 200]
    scale = (0.02, 1.0)
    ratio = (3. / 4, 4. / 3)
    config = {
        'size': size,
        'scale': scale,
        'ratio': ratio,
        'interpolation': None
    }

    expected_size = [200, 200, 3]

    for i in self.INTERPOLATION_METHODS:
      config['interpolation'] = i
      output = augmentations.RandomCropThenResize(config)(self.image)
      self.assertEqual(output.shape, expected_size)

    # case scale less than 0 and greater than 1
    scales = [[-1.01, 1.0], [0.01, 1.5]]  # negative tests
    for i in scales:
      config['scale'] = i
      config['interpolation'] = 'nearest'
      with self.assertRaises(InvalidArgumentError):
        output = augmentations.RandomCropThenResize(config)(self.image)

    # case ratio not less than 0
    config['ratio'] = (0, 1.33)
    config['scale'] = scale
    with self.assertRaises(InvalidArgumentError):
      output = augmentations.RandomCropThenResize(config)(self.image)

  def test_randomcrop(self):
    size = [100, 50, 3]
    config = {'size': size}

    # case 1
    output = augmentations.RandomCrop(config)(self.image)
    self.assertEqual(output.shape, (100, 50, 3))

    # case 2 negative case
    size = [300, 300, 3]
    config = {'size': size}
    with self.assertRaises(InvalidArgumentError):
      output = augmentations.RandomCrop(config)(self.image)

  def test_centralcrop(self):
    size = [100, 50]
    crop_proportion = 0.875
    config = {
      'size': size,
      'crop_proportion': crop_proportion,
      'interpolation': None
    }

    # case 1
    for i in self.INTERPOLATION_METHODS:
      config['interpolation'] = i
      output = augmentations.CentralCrop(config)(self.image)
      self.assertEqual(output.shape, (100, 50, 3))

    # case 2 interpolation not found in the list
    with self.assertRaises(ValueError):
      config['interpolation'] = 'test'
      output = augmentations.CentralCrop(config)(self.image)

    # case 3 crop_proportion less than and greater than 1
    crop_proportions = [-0.01, 1.01]
    config = {
      'size': size,
      'crop_proportion': None,
      'interpolation': 'bicubic'
    }
    for i in crop_proportions:
      config['crop_proportion'] = i
      with self.assertRaises(InvalidArgumentError):
        output = augmentations.CentralCrop(config)(self.image)

  def test_random_horizontal_vertical_flip_rotate_90(self):
    sum_pixels = np.sum(self.rand_image)

    output = [
        augmentations.RandomHorizontalFlip({})(self.rand_image),
        augmentations.RandomVerticalFlip({})(self.rand_image),
        augmentations.RandomRotate90({})(self.rand_image)
    ]

    for i in output:
      # check pixel are only shifted and not changed
      self.assertEqual(np.sum(i), sum_pixels)
      # check if the size remains the same
      self.assertEqual(i.shape, [224, 224, 3])

  def test_colorjitter(self):
    # case 1 value is zero
    brightness = contrast = saturation = hue = 0.0
    config = {
      'brightness': brightness,
      'contrast': (contrast, contrast),
      'saturation': (saturation, saturation),
      'hue': hue,
      'clip': (0., 0.),
    }
    output = augmentations.ColorJitter(config)(self.image)
    self.assertTrue(np.allclose(output, self.image, rtol=1e-4, atol=1e-4))
    self.assertTrue(output.shape, self.image.shape)

    # case 2 positive scenario defaults
    config = {
      'brightness': 0.05,
      'contrast': (0.7, 1.3),
      'saturation': (0.6, 1.6),
      'hue': 0.08,
      'clip': (0., 0.),
    }
    output = augmentations.ColorJitter(config)(self.image)
    self.assertTrue(np.allclose(output, self.image, rtol=1e-2, atol=1e-2))
    self.assertTrue(output.shape, self.image.shape)

    # case 3 clip value higher than 255.
    config = {
      'brightness': 0.05,
      'contrast': (0.7, 1.3),
      'saturation': (0.6, 1.6),
      'hue': 0.08,
      'clip': (0., 255.),
    }
    output = augmentations.ColorJitter(config)(self.image)
    self.assertTrue(np.allclose(output, self.image, rtol=1e-2, atol=1e-2))
    self.assertTrue(output.shape, (self.image).shape)

  def test_random_rotate(self):
    angle = 10
    config = {
      'angle': angle,
      'interpolation': 'nearest'
    }
    output = augmentations.RandomRotate(config)(self.image)
    if tf.is_tensor(output):
      output = output.numpy()
    # output and input shape should be same
    self.assertEqual(self.image.shape, output.shape)

    num_of_times_rot_not_applied = 0
    total_trial = 100
    for i in range(total_trial):
      temp_output = augmentations.RandomRotate(config)(self.image)
      if tf.is_tensor(temp_output):
        temp_output = temp_output.numpy()
      # if input and output image sum is equal, means rotation is not applied
      if np.sum(self.image) == np.sum(temp_output):
        num_of_times_rot_not_applied += 1
    # Check if input image is unchanged few times
    self.assertTrue(num_of_times_rot_not_applied > 0)

    # Check if input image is changed because of rotation
    if augmentations.get_tfa_available():
      self.assertTrue((total_trial - num_of_times_rot_not_applied) > 0)

  def test_translate(self):
    padding_stragies = ["CONSTANT", "REFLECT", "SYMMETRIC"] 
    width_shift_range = 0.1
    height_shift_range = 0.1
    for strategy in padding_stragies:
      config = {
        'width_shift_range': width_shift_range,
        'height_shift_range': height_shift_range,
        'padding_strategy': strategy
      }
      output = augmentations.Translate(config)(self.image)
      if tf.is_tensor(output):
        output = output.numpy()
      # output and input shape should be same
      self.assertEqual(self.image.shape, output.shape)
      # Count number of rows with zeros (which will be regarded as the number of pixel shifts either
      row_counts = len(np.where(~output.any(axis=0))[0]) / 3.
      col_counts = len(np.where(~output.any(axis=1))[0]) / 3.
      # Check whether translation applied in the provided range
      self.assertTrue(0 <= row_counts / self.image.shape[0] <= width_shift_range)
      self.assertTrue(0 <= col_counts / self.image.shape[1] <= height_shift_range)

    # negative test - invalid padding strategy
    config = {
      'width_shift_range': width_shift_range,
      'height_shift_range': height_shift_range,
      'padding_strategy': 'test'
    }
    with self.assertRaises(AssertionError):
      output = augmentations.Translate(config)(self.image)

  def test_cutout(self):
    config = {
      'length': 16
    }
    tf.random.set_seed(42)
    image = tf.ones((224, 224, 3)) * 255.
    image = augmentations.Cutout(config)(image)
    image = tf.cast(image, tf.uint8)
    self.assertEqual(image.numpy().sum(), ((224**2-16**2) * 3)*255)
    print(image.numpy().shape)
    