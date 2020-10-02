import glob
import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np
import tensorflow as tf
from . import dataloader


class TestDataLoader(unittest.TestCase):
  def test_map_fn(self):
    transformation_list = ['ResizeThenRandomCrop']
    config = {
      'ResizeThenRandomCrop': {
        "size": [256, 256],
        "crop_size": [224, 224, 3],
        "interpolation": 'bicubic'
      }
    }
    map_fn = dataloader.get_map_fn(transformation_list, config, n_classes=2)
    dataset_dir = create_fake_dataset()
    image = cv2.imread(os.path.join(dataset_dir, 'dog/1.jpg'))
    image, label = map_fn(tf.convert_to_tensor(image), tf.convert_to_tensor(1))
    self.assertEqual(label.numpy()[0], 0)
    self.assertEqual(label.numpy()[1], 1)
    self.assertTrue(np.array_equal(image.numpy(), np.ones((224, 224, 3) , dtype=np.float32)*255))

  def test_get_dataset_from_tfds(self):
    config = {
      'name': 'mnist',
      'data_dir': None,
      'batch_size': 7,
      'train_split_id': 'train'
    }
    dataset = dataloader.get_dataset_from_tfds(config, [], 10, split='train')

    i = 0
    for image, label in dataset:
      self.assertEqual(label.numpy().shape, (7, 10))
      self.assertTrue(label.numpy()[0, 0] in [0, 1])
      self.assertTrue(label.numpy()[1, 1] in [0, 1])
      self.assertEqual(image.numpy().shape, (7, 28, 28, 1))
      i += 1
      if i == 3:
        break

    self.assertEqual(i, 3)


def create_fake_dataset(n_images_per_class=2):
  dataset_dir = tempfile.mkdtemp()
  os.makedirs(os.path.join(dataset_dir, 'cat'))
  os.makedirs(os.path.join(dataset_dir, 'dog'))
  for i in range(n_images_per_class):
    cv2.imwrite(os.path.join(dataset_dir, 'dog', '{}.jpg'.format(i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
    cv2.imwrite(os.path.join(dataset_dir, 'cat', '{}.jpg'.format(i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
  return dataset_dir
