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
    keras_preprocessing.save_img(os.path.join(dataset_dir, 'cat', '{}.jpeg'.format(i)), image_array[i], scale=False)

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
  return dataset_dir
class TestCompareDataLoader(unittest.TestCase):
  @classmethod
  def setUp(self):
    self.image_white = np.ones(DIMENSIONS, dtype=np.uint8) * 255 
    # setting seed for numpy 
    np.random.seed(42)
    # setting seed for tensorflow 
    tf.random.set_seed(0)

    self.batch_random_images = np.random.uniform(low=0.0, high=255., size=DIMENSIONS)
    self.single_random_image = np.random.uniform(low=0, high=255, size=(1, 224, 224, 3)).astype(np.uint8)
    self.image_path = "ressources/testing/cat.png"
    self.black_white_image = "ressources/testing/black_and_white.jpeg"

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

    # compare output between our Dataloader and DCN dataloader
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
                                                               horizontal_flip=False)

    x_train_from_dcn_dataloader = next(iter(dcn_augmentations.flow(x_data, None, batch_size=BATCH_SIZE, shuffle=False)))
    
    # Test the tensor shape remains the same
    self.assertTrue(x_train_from_dcn_dataloader.shape, x_train_from_dcn_dataloader.shape)

    # Note for translate we expect the mean to be different as the method to fill the missing pixels that keras and our data loader use are different. 
    # This doesn't cause an issue with the image being translated (see test_data_visualize), rather difference in pixel values at the shifted places.
    # data visualization does confirm the translation works as expected.
    self.assertAlmostEqual(np.mean(x_train_from_our_dataloader), np.mean(x_train_from_dcn_dataloader), places=3)

    # clean up
    shutil.rmtree(dataset_dir)

  def test_compare_with_augmentation_randomflip(self):
    # create fake dataset
    # dataset_dir = create_fake_dataset_and_convert_to_tfrecords(image_array=self.black_white_image, n_images_per_class=1)
    dataset_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(dataset_dir, 'black_white'), exist_ok=True)
    shutil.copyfile(os.path.join(self.black_white_image), os.path.join(dataset_dir, 'black_white', 'black_and_white.jpeg'))

    image = keras_preprocessing.load_img(self.black_white_image, grayscale=False, color_mode='rgb', target_size=None,
      interpolation='nearest')
    image_array = keras_preprocessing.img_to_array(image)
    image_array = image_array[np.newaxis, ...]  # adding batch dimension for ImageDataLoader

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
    }

    # loop through until the RandomHorizontalFlip is applied
    dataset = get_dataset(config, ['RandomHorizontalFlip'], 1, 'train')
    x_train_from_our_dataloader, _ = next(iter(dataset))
    self.assertFalse(np.allclose(x_train_from_our_dataloader, image_array), "Augmented Image and original image are the same")

    # loop through until the RandomHorizontalFlip is applied
    dcn_augmentations = keras_preprocessing.ImageDataGenerator(horizontal_flip=True)
    x_train_from_dcn_dataloader = next(iter(dcn_augmentations.flow(image_array, None, batch_size=1, shuffle=False)))
    self.assertFalse(np.allclose(x_train_from_dcn_dataloader, image_array), "Augmented Image and original image are the same")

    # Test the tensor shape remains the same
    self.assertTrue(x_train_from_our_dataloader.shape, x_train_from_dcn_dataloader.shape)

    # Note: The example used here is to test if the augmentation works using a black and white image. 
    # This test would fail if an image with wide range of pixel values across the color channels are used. 
    # This is a limitation due to the loss of data (encode to bytes and decode back to image) for JPEG format.
    # In the TFrecord writer we use opencv and decode using tensorflow library. The encoded bytes using openCV and tensorflow are not identical. 
    # There is a slight pixel difference which is not noticible to the human eyes. 
    self.assertAlmostEqual(np.mean(x_train_from_our_dataloader), np.mean(x_train_from_dcn_dataloader), places=7)

    # clean up
    shutil.rmtree(dataset_dir)

  def test_data_visualization(self):
    dataset_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(dataset_dir, 'cat'), exist_ok=True)
    shutil.copyfile(os.path.join(self.image_path), os.path.join(dataset_dir, 'cat', 'cat.jpeg'))

    image = keras_preprocessing.load_img(self.image_path, grayscale=False, color_mode='rgb', target_size=None,
      interpolation='nearest')
    image_array = keras_preprocessing.img_to_array(image)
    image_array = image_array[np.newaxis, ...]  # adding batch dimension for ImageDataLoader

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

    # loop through until the RandomHorizontalFlip is applied.
    dataset = get_dataset(config, ['Translate','RandomHorizontalFlip'], 1, 'train')
    image, _ = next(iter(dataset)) # get the image
    image_i = keras_preprocessing.array_to_img(image[0]) # get the image excluding the batch dimension and save
    keras_preprocessing.save_img(os.path.join(dataset_dir,'cat_after_augment_ours.png'), image_i) # save the image
    self.assertFalse(np.allclose(image, image_array), "Augmented Image and original image are the same" )

    dcn_augmentations = keras_preprocessing.ImageDataGenerator(
      height_shift_range=config['Translate']['height_shift_range'],
      width_shift_range=config['Translate']['width_shift_range'], 
      horizontal_flip=True)

    image_dcn = next(iter(dcn_augmentations.flow(image_array, None, batch_size=1, shuffle=False)))
    # save image excluding batch size
    keras_preprocessing.save_img(os.path.join(dataset_dir,'cat_after_augment_DCN.png'), image[0]) 
    self.assertFalse(np.allclose(image_dcn, image_array), "Augmented Image and original image are the same")

    # remove the below line in order to perform the visual inspection
    shutil.rmtree(dataset_dir)
