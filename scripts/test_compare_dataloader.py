""" 
Known differences: 

1) Our augmentation strategy works by taking mean by stacking each color channel whereas
DCN takes each pixel wise mean

2) There is very small difference in the mean between Keras Transate and our version.

"""
import unittest
import tempfile
import os
import shutil

import numpy as np
import cv2

import tensorflow as tf 

from tfrecord_writer import build_tfrecord_dataset
from src.data.dataloader import get_dataset
from src.data.augmentations import Normalize, Translate

class TestCompareDataLoader(unittest.TestCase):
  @classmethod
  def setUp(self):
    self.image_ones = np.ones((20, 224, 224, 3), dtype=np.uint8) * 255
    np.random.seed(42)
    self.image_random = np.random.uniform(low=0.0, high=255., size=(20, 224, 224, 3))

  def get_dcn_dataloader(self, image_array):
    """DCN data loader from the DCN training file. takes input as a image array, normalizes the input and returns it
    Note: some parts are commented as its not required for this specific tests. 
    Args:
        image_array (numpy array): image numpy ndarray. 

    Returns:
        [numpy array]: Normalized output
    """
    # (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train) = image_array, np.ones((20, 2)) 
    nb_classes                           = 2
    n_train                              = 20
    shuf_inds  = np.arange(len(y_train))
    # np.random.seed(0xDEADBEEF)
    np.random.shuffle(shuf_inds)
    train_inds = shuf_inds[:n_train]
    # val_inds   = shuf_inds[n_train:]

    X_train    = X_train.astype('float32')  /255.0
    # X_test     = X_test .astype('float32')/255.0

    X_train_split = X_train[train_inds]
    # X_val_split   = X_train[val_inds  ]
    y_train_split = y_train[train_inds]
    # y_val_split   = y_train[val_inds  ]

    pixel_mean = np.mean(X_train_split, axis=0)
    # print(pixel_mean)

    X_train    = X_train_split.astype(np.float32) - pixel_mean
    # X_val      = X_val_split  .astype(np.float32) - pixel_mean
    # X_test     = X_test       .astype(np.float32) - pixel_mean

    # Y_train    = tf.keras.utils.to_categorical(y_train_split, nb_classes)
    # Y_val      = tf.keras.utils.to_categorical(y_val_split,   nb_classes)
    # Y_test     = tf.keras.utils.to_categorical(y_test,        nb_classes)

    return X_train # Y_train, X_val, Y_val

  def create_fake_dataset_and_convert_to_tfrecords(self, image_array, n_images_per_class=2):
    dataset_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(dataset_dir, 'cat'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'dog'), exist_ok=True)
    for i in range(n_images_per_class):
      cv2.imwrite(os.path.join(dataset_dir, 'dog', '{}.jpg'.format(i)), image_array[i])
      cv2.imwrite(os.path.join(dataset_dir, 'cat', '{}.jpg'.format(i)), image_array[i+1])
    
    args = {'name': 'tfrecords', 'description': 'test', 'tfrecord_dir_path': dataset_dir,
      'tfrecord_size': 1, 'preprocessing': 'NO', 'image_size': (224, 224), "n_tfrecords":1,
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

  def test_compare_without_augmentations(self):
    # create fake dataset and convert TFrecords
    dataset_dir = self.create_fake_dataset_and_convert_to_tfrecords(image_array=self.image_ones, n_images_per_class=10)

    config = {
      'name': 'tfrecords',
      'data_dir': dataset_dir,
      'batch_size': 20,
      'train_split_id': 'train',
    }
    
    # get data from tfrecord
    dataset = get_dataset(config, [], 2, 'train')
    for image, _  in dataset:
      # Normalize the image similar to DCN strategy
      normalize = image / 255.
      subtract_mean = tf.reduce_mean(normalize, axis=0)
      X_train_from_our_dataloader = normalize - subtract_mean

    
    print(np.min(X_train_from_our_dataloader), np.max(X_train_from_our_dataloader))
    X_train_from_dcn_dataloader = self.get_dcn_dataloader(self.image_ones)
    print(np.min(X_train_from_dcn_dataloader), np.max(X_train_from_dcn_dataloader))

    self.assertEqual(np.allclose(X_train_from_our_dataloader, X_train_from_dcn_dataloader), True)

    # clean up
    shutil.rmtree(dataset_dir)

  def test_compare_without_augmentations_random(self):
    # create fake dataset and convert TFrecords
    dataset_dir = self.create_fake_dataset_and_convert_to_tfrecords(image_array=self.image_random, n_images_per_class=10)

    # Our augmentation strategy works by taking mean by stacking each color channel whereas DCN takes each pixel wise mean 
    # Not sure if this difference can cause changes to the final accuracy. 

    # image_normalized = self.image_random / 255.
    # r = np.dstack([image_normalized[i][:, :, 0] for i in range(len(image_normalized))])
    # g = np.dstack([image_normalized[i][:, :, 1] for i in range(len(image_normalized))])
    # b = np.dstack([image_normalized[i][:, :, 2] for i in range(len(image_normalized))]) 
    # mean_data = [np.mean(r), np.mean(g), np.mean(b)]
    # print(mean_data)

    config = {
      'name': 'tfrecords',
      'data_dir': dataset_dir,
      'batch_size': 20,
      'train_split_id': 'train',
      # 'Normalize': {'only_subtract_mean': True,
      # 'scale_in_zero_to_one': True,
      # 'mean': mean_data, 
      # 'std': [0, 0, 0]
      # }
    }
    
    # get data from tfrecord
    dataset = get_dataset(config, [], 2, 'train')
    for image, _  in dataset:
      # Normalize the image similar to DCN strategy
      normalize = image / 255.
      subtract_mean = tf.reduce_mean(normalize, axis=0)
      X_train_from_our_dataloader = normalize - subtract_mean

    # get x_train from dcn 
    X_train_from_dcn_dataloader = self.get_dcn_dataloader(self.image_random)

    self.assertAlmostEqual(np.mean(X_train_from_our_dataloader), np.mean(X_train_from_dcn_dataloader), places=5)

    # clean up
    shutil.rmtree(dataset_dir)

  def test_compare_with_translate_random(self):
    # create fake dataset
    dataset_dir = self.create_fake_dataset_and_convert_to_tfrecords(image_array=self.image_random, n_images_per_class=10)

    image_normalized = self.image_random / 255.
    r = np.dstack([image_normalized[i][:, :, 0] for i in range(len(image_normalized))])
    g = np.dstack([image_normalized[i][:, :, 1] for i in range(len(image_normalized))])
    b = np.dstack([image_normalized[i][:, :, 2] for i in range(len(image_normalized))]) 
    mean_data = [np.mean(r), np.mean(g), np.mean(b)]
    print(mean_data)

    config = {
      'name': 'tfrecords',
      'data_dir': dataset_dir,
      'batch_size': 20,
      'train_split_id': 'train',
      'Translate': {
      'width_shift_range': 0.125,
      'height_shift_range': 0.125
      }
    }
    
    dataset = get_dataset(config, ['Translate', 'RandomHorizontalFlip'], 2, 'train')
    for image, _  in dataset:
      # Normalize the image similar to DCN strategy
      normalize = image / 255.
      subtract_mean = tf.reduce_mean(normalize, axis=0)
      X_train_from_our_dataloader = normalize - subtract_mean

    # get x_train from dcn 
    x_data = self.get_dcn_dataloader(self.image_random)

    # DCN augmentation strategy
    dcn_augmentations = tf.keras.preprocessing.image.ImageDataGenerator(
                                                                    height_shift_range=config['Translate']['height_shift_range'],
                                                                    width_shift_range=config['Translate']['width_shift_range'],
                                                                    horizontal_flip=True
                                                                    )

    for image in dcn_augmentations.flow(x_data, None, batch_size=20, shuffle=False):
      X_train_from_dcn_dataloader = image
      break
    
    # Test the tensor shape remains the same
    self.assertTrue(X_train_from_dcn_dataloader.shape, X_train_from_dcn_dataloader.shape)
    print(np.mean(X_train_from_our_dataloader), np.mean(X_train_from_dcn_dataloader))

    self.assertAlmostEqual(np.mean(X_train_from_our_dataloader), np.mean(X_train_from_dcn_dataloader), places=3)

    # clean up
    shutil.rmtree(dataset_dir)