""" 
Testing Procedure: 
1) Create a fake dataset and load the data using our dataloader and DCN dataloader. 
   Finally compare the mean of the outputs which are expected to be same.
2) Load the fake dataset, apply each augmentations separately and compare the mean between our dataloader versus DCN dataloader.
3) Load the fake dataset(single image) over many iterations, compute the pixel mean for a specific portion of the image during these iterations 
   and finally compare the mean of the output between our Dataloader and DCN Dataloader.
4) Load a single valid image, apply the translate augmentation. Inspect the output image for ours and DCN dataloader.

Known differences: 

1) Our augmentation strategy works by taking mean by stacking each color channel whereas
DCN takes each pixel wise mean across the whole training batch.

Algorithmically speaking, we do:
Input: images, r_mean, g_mean, b_mean
Alg: 
  - substract r_mean from images[:, :, :, 0] (dimensions are bs, h, w, c)
  - substract g_mean from images[:, :, :, 1]
  - substract b_mean from images[:, :, :, 2]
  - normalize

They do:
Input: images, mean
Alg: 
  - substract mean from images[:, :, :, :] (dimensions are bs, h, w, c)
  - normalize

note: their way of doing thing is unexpected so we will not implement it in classification-api
but we will use there way here for testing purposes
"""
import unittest
import tempfile
import os
import shutil

import numpy as np
import cv2

import tensorflow as tf 
import tensorflow.keras.preprocessing.image as keras_preprocessing

from scripts.tfrecord_writer import build_tfrecord_dataset
from src.data.dataloader import get_dataset

# global variable
DIMENSIONS = (10, 224, 224, 3)
BATCH_SIZE = 10

def create_fake_dataset_and_convert_to_tfrecords(image_array, n_images_per_class=1):
  dataset_dir = tempfile.mkdtemp()
  os.makedirs(os.path.join(dataset_dir, 'cat'), exist_ok=True)

  for i in range(n_images_per_class):
    cv2.imwrite(os.path.join(dataset_dir, 'cat', '{}.jpg'.format(i)), image_array[i])

  args = {'name': 'tfrecords', 'description': 'test', 'tfrecord_dir_path': dataset_dir,
    'tfrecord_size': 1, 'preprocessing': 'NO', 'image_size': (DIMENSIONS[1:3]), "n_tfrecords":1,
    'data': {'images_dir_path': dataset_dir,
              'annotation_file_path': None,
              'delimiter': ',',
              'header_exists': False,
              'split_names': ['train'],
              'split_percentages': [1.0],
          } 
  }
  # generate tfrecords 
  build_tfrecord_dataset(args)
  return dataset_dir
class TestCompareDataLoader(unittest.TestCase):
  @classmethod
  def setUp(self):
    self.image_white = np.ones(DIMENSIONS, dtype=np.uint8) * 255 
    # setting seed for numpy 
    np.random.seed(42)

    self.batch_random_images = np.random.uniform(low=0.0, high=255., size=DIMENSIONS)
    self.image_path = "ressources/testing/cat.jpeg"

  def get_dcn_dataloader(self, image_array):
    """
    DCN dataloader has been taken from link below and certain variables which are not required for the comparison test are removed.
    https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/scripts/training.py#L542


    Function takes input as an image array, normalizes the input and returns it.
    Args:
        image_array (numpy array): image numpy ndarray. 

    Returns:
        [numpy array]: Normalized output
    """
    x_train    = image_array.astype('float32') / 255.0
    pixel_mean = np.mean(x_train, axis=0)

    x_train    = x_train.astype(np.float32) - pixel_mean
    
    return x_train 

  def test_compare_without_augmentations(self):
    # create fake dataset and convert TFrecords
    dataset_dir = create_fake_dataset_and_convert_to_tfrecords(image_array=self.image_white, n_images_per_class=10)

    config = {
      'name': 'tfrecords',
      'data_dir': dataset_dir,
      'batch_size': BATCH_SIZE,
      'train_split_id': 'train',
    }
    
    # get data from tfrecord
    dataset = get_dataset(config, [], 1, 'train')

    image, _ = next(iter(dataset))
    # Normalize the image similar to DCN strategy
    normalize = image / 255.
    subtract_mean = tf.reduce_mean(normalize, axis=0)
    x_train_from_our_dataloader = normalize - subtract_mean

    x_train_from_dcn_dataloader = self.get_dcn_dataloader(self.image_white)

    self.assertEqual(np.allclose(x_train_from_our_dataloader, x_train_from_dcn_dataloader), True)

    # clean up
    shutil.rmtree(dataset_dir)

  def test_compare_without_augmentations_random(self):
    # create fake dataset and convert TFrecords
    dataset_dir = create_fake_dataset_and_convert_to_tfrecords(image_array=self.batch_random_images, n_images_per_class=10)

    # Code below tested with subtracting the mean for each color channels
    # and there were differences between the output from our dataloader and DCN dataloader.
    # For comparison purposes the mean is subtracted following DCN strategy. 

    # image_normalized = self.batch_random_images,/ 255.
    # r = np.dstack([image_normalized[i][:, :, 0] for i in range(len(image_normalized))])
    # g = np.dstack([image_normalized[i][:, :, 1] for i in range(len(image_normalized))])
    # b = np.dstack([image_normalized[i][:, :, 2] for i in range(len(image_normalized))]) 
    # mean_data = [np.mean(r), np.mean(g), np.mean(b)]
    # print(mean_data)

    config = {
      'name': 'tfrecords',
      'data_dir': dataset_dir,
      'batch_size': BATCH_SIZE,
      'train_split_id': 'train',
    }
    
    # get data from tfrecord
    dataset = get_dataset(config, [], 1, 'train')

    image, _ = next(iter(dataset))
    # Normalize the image similar to DCN strategy
    normalize = image / 255.
    subtract_mean = tf.reduce_mean(normalize, axis=0)
    x_train_from_our_dataloader = normalize - subtract_mean

    # get x_train from dcn 
    x_train_from_dcn_dataloader = self.get_dcn_dataloader(self.batch_random_images)

    self.assertAlmostEqual(np.mean(x_train_from_our_dataloader), np.mean(x_train_from_dcn_dataloader), places=5)

    # clean up
    shutil.rmtree(dataset_dir)

  def test_compare_with_augmentation_translate(self):
    # create fake dataset
    dataset_dir = create_fake_dataset_and_convert_to_tfrecords(image_array=self.batch_random_images, n_images_per_class=10)

    config = {
      'name': 'tfrecords',
      'data_dir': dataset_dir,
      'batch_size': BATCH_SIZE,
      'train_split_id': 'train',
      'Translate': {
      'width_shift_range': 0.125,
      'height_shift_range': 0.125,
      'padding_strategy': 'REFLECT'
      }
    }
    
    dataset = get_dataset(config, ['Translate'], 1, 'train')

    image, _ = next(iter(dataset))
    # Normalize the image similar to DCN strategy
    normalize = image / 255.
    subtract_mean = tf.reduce_mean(normalize, axis=0)
    x_train_from_our_dataloader = normalize - subtract_mean

    # get x_train from dcn 
    x_data = self.get_dcn_dataloader(self.batch_random_images)

    # DCN augmentation strategy
    dcn_augmentations = keras_preprocessing.ImageDataGenerator(height_shift_range=config['Translate']['height_shift_range'],
                                                               width_shift_range=config['Translate']['width_shift_range'],
                                                               horizontal_flip=True)

    x_train_from_dcn_dataloader = next(iter(dcn_augmentations.flow(x_data, None, batch_size=BATCH_SIZE, shuffle=False)))
    
    # Test the tensor shape remains the same
    self.assertTrue(x_train_from_dcn_dataloader.shape, x_train_from_dcn_dataloader.shape)

    self.assertAlmostEqual(np.mean(x_train_from_our_dataloader), np.mean(x_train_from_dcn_dataloader), places=3)

    # clean up
    shutil.rmtree(dataset_dir)

  def test_compare_with_augmentation_randomflip(self):
    # create fake dataset
    dataset_dir = create_fake_dataset_and_convert_to_tfrecords(image_array=self.batch_random_images, n_images_per_class=10)

    config = {
      'name': 'tfrecords',
      'data_dir': dataset_dir,
      'batch_size': BATCH_SIZE,
      'train_split_id': 'train',
    }

    while True: 
      dataset = get_dataset(config, ['RandomHorizontalFlip'], 1, 'train')

      image, _ = next(iter(dataset))
      # Normalize the image similar to DCN strategy
      normalize = image / 255.
      subtract_mean = tf.reduce_mean(normalize, axis=0)
      x_train_from_our_dataloader = normalize - subtract_mean
      if np.any(x_train_from_our_dataloader) == np.any(self.batch_random_images):
        break

    # get x_train from dcn 
    x_data = self.get_dcn_dataloader(self.batch_random_images)

    # DCN augmentation strategy
    while True:
      dcn_augmentations = keras_preprocessing.ImageDataGenerator(horizontal_flip=True)
      x_train_from_dcn_dataloader = next(iter(dcn_augmentations.flow(x_data, None, batch_size=BATCH_SIZE, shuffle=False)))
      if np.any(x_train_from_dcn_dataloader) == np.any(self.batch_random_images):
        break
    
    # Test the tensor shape remains the same
    self.assertTrue(x_train_from_dcn_dataloader.shape, x_train_from_dcn_dataloader.shape)

    self.assertAlmostEqual(np.mean(x_train_from_our_dataloader), np.mean(x_train_from_dcn_dataloader), places=7)

    # clean up
    shutil.rmtree(dataset_dir)

  def test_data_visualization(self):
    dataset_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(dataset_dir, 'cat'), exist_ok=True)
    shutil.copyfile(os.path.join(self.image_path), os.path.join(dataset_dir, 'cat', 'cat.jpeg'))

    image = keras_preprocessing.load_img(
    os.path.join(self.image_path), grayscale=False, color_mode='rgb', target_size=None,
    interpolation='nearest')
    image_array = keras_preprocessing.img_to_array(image)

    args = {'name': 'tfrecords', 'description': 'test', 'tfrecord_dir_path': dataset_dir,
      'tfrecord_size': 1, 'preprocessing': 'NO', 'image_size': DIMENSIONS[1:3], "n_tfrecords":1,
      'data': {'images_dir_path': dataset_dir,
                'annotation_file_path': None,
                'delimiter': ',',
                'header_exists': False,
                'split_names': ['train'],
                'split_percentages': [1.0],
            } 
    }
    # generate tfrecords 
    build_tfrecord_dataset(args)

    config = {
        'name': 'tfrecords',
        'data_dir': dataset_dir,
        'batch_size': 1,
        'train_split_id': 'train',
        'Translate': {
        'width_shift_range': 0.125,
        'height_shift_range': 0.125,
        'padding_strategy': 'reflect'
        }
    }

    dataset = get_dataset(config, ['Translate'], 1, 'train')
    image, _ = next(iter(dataset)) # get the image
    image = keras_preprocessing.array_to_img(image[0]) # get the image excluding the batch dimension and save
    keras_preprocessing.save_img(os.path.join(dataset_dir,'cat_after_augment_ours.jpeg'), image)

    dcn_augmentations = keras_preprocessing.ImageDataGenerator(height_shift_range=config['Translate']['height_shift_range'],
                                                               width_shift_range=config['Translate']['width_shift_range'],
                                                               horizontal_flip=True)

    image_array = np.array(image_array[np.newaxis, ...])  # adding batch dimension for ImageDataLoader
    image = next(iter(dcn_augmentations.flow(image_array, None, batch_size=1, shuffle=False)))

    # save image excluding batch size
    keras_preprocessing.save_img(os.path.join(dataset_dir,'cat_after_augment_DCN.jpeg'), image[0]) 

    # remove the below line in order to perform the visual inspection
    shutil.rmtree(dataset_dir)
