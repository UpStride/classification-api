import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import tensorflow as tf

from .data import dataloader
from .data.test_dataloader import create_fake_dataset
from .models.generic_model import GenericModel
from .utils import (init_custom_checkpoint_callbacks, copy_and_resize, get_imagenet_data, get_partial_paths,
                    get_paths, get_synset, get_val_label_dict)


class Model1(GenericModel):
  def model(self):
    self.x = self.layers().Flatten()(self.x)
    self.x = self.layers().Dense(10)(self.x)


class TestUtils(unittest.TestCase):
  def test_copy_and_resize(self):
    source = create_fake_dataset()
    dest = tempfile.mkdtemp()
    copy_and_resize(source, dest, 256)
    self.assertEqual(sorted(os.listdir(dest)), ['cat', 'dog'])
    self.assertEqual(sorted(os.listdir(os.path.join(dest, 'cat'))), ['0.jpg', '1.jpg'])
    self.assertEqual(sorted(os.listdir(os.path.join(dest, 'dog'))), ['0.jpg', '1.jpg'])
    self.assertEqual(cv2.imread(os.path.join(dest, 'dog', '1.jpg')).shape[0], 256)
    shutil.rmtree(source)
    shutil.rmtree(dest)

  def test_copy_and_resize_final_slash(self):
    source = create_fake_dataset()
    source += '/'
    dest = tempfile.mkdtemp()
    copy_and_resize(source, dest, 256)
    self.assertEqual(sorted(os.listdir(dest)), ['cat', 'dog'])
    self.assertEqual(sorted(os.listdir(os.path.join(dest, 'cat'))), ['0.jpg', '1.jpg'])
    self.assertEqual(sorted(os.listdir(os.path.join(dest, 'dog'))), ['0.jpg', '1.jpg'])
    self.assertEqual(cv2.imread(os.path.join(dest, 'dog', '1.jpg')).shape[0], 256)
    shutil.rmtree(source)
    shutil.rmtree(dest)

  def test_get_synset(self):
    synset = get_synset("ressources/testing/fake_LOC_synset_mapping.txt")
    self.assertEqual(synset['n01496331'], 5)

  def test_get_paths(self):
    training_dataset = create_fake_training_data()
    val_dataset = create_fake_val_data()
    training_images = get_paths(training_dataset)
    val_images = get_paths(val_dataset)
    self.assertEqual(len(training_images), 20)
    self.assertEqual(len(val_images), 2)

  def test_get_val_label_dict(self):
    val_dict = get_val_label_dict("ressources/testing/fake_LOC_val_solution.csv")
    self.assertEqual(val_dict['ILSVRC2012_val_0'], 'n01484850')

  def test_get_partial_paths(self):
    training = create_fake_training_data()
    paths = get_partial_paths(training, 50)
    self.assertEqual(len(paths), 10)
    paths = get_partial_paths(training, 100)
    self.assertEqual(len(paths), 20)
    paths = get_partial_paths(training, 40)
    self.assertEqual(len(paths), 0)

  def test_get_imagenet_data(self):
    train_dir = create_fake_training_data()
    val_dir = create_fake_val_data()
    synset_path = "ressources/testing/fake_LOC_synset_mapping.txt"
    training_percentage = 100
    val_gt_path = "ressources/testing/fake_LOC_val_solution.csv"
    imagenet_data = get_imagenet_data({'synset_path': synset_path,
                                       'train_dir': train_dir,
                                       'train_data_percentage': training_percentage,
                                       'val_dir': val_dir,
                                       'val_gt_path': val_gt_path})
    train_paths, train_labels, val_paths, val_labels = imagenet_data
    self.assertEqual(type(train_paths[0]), str)
    self.assertEqual(type(val_paths[0]), str)
    self.assertEqual(type(train_labels[0]), int)
    self.assertEqual(type(val_labels[0]), int)
    self.assertEqual(len(train_paths), 20)
    self.assertEqual(len(train_labels), 20)
    self.assertEqual(len(val_paths), 2)
    self.assertEqual(len(val_labels), 2)

  def test_data_pipeline_imagenet_data(self):
    train_dir = create_fake_training_data()
    val_dir = create_fake_val_data()
    synset_path = "ressources/testing/fake_LOC_synset_mapping.txt"
    training_percentage = 100
    val_gt_path = "ressources/testing/fake_LOC_val_solution.csv"
    imagenet_data = get_imagenet_data({'synset_path': synset_path,
                                       'train_dir': train_dir,
                                       'train_data_percentage': training_percentage,
                                       'val_dir': val_dir,
                                       'val_gt_path': val_gt_path})
    train_paths, train_labels, val_paths, val_labels = imagenet_data
    dataset = dataloader.get_dataset(train_paths, train_labels, n_classes=10, batch_size=2)
    i = 0
    for image, label in dataset:
      self.assertEqual(label.numpy().shape, (2, 10))
      self.assertTrue(label.numpy()[0, 0] in [0, 1])
      self.assertTrue(label.numpy()[1, 1] in [0, 1])
      self.assertEqual(image.numpy().shape, (2, 224, 224, 3))
      i += 1

    self.assertEqual(i, 10)

    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)

  def test_init_custom_checkpoint_callbacks(self):
    model = Model1('tensorflow', factor=1).model
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    ckpt_dir = tempfile.mkdtemp()
    callback, latest_epoch = init_custom_checkpoint_callbacks(model, optimizer, ckpt_dir, max_ckpt=5, save_frequency=1)
    self.assertEqual(os.listdir(ckpt_dir), [])
    self.assertEqual(latest_epoch, 0)

    # train for one epoch
    train_dir = create_fake_training_data()
    val_dir = create_fake_val_data()
    synset_path = "ressources/testing/fake_LOC_synset_mapping.txt"
    training_percentage = 100
    val_gt_path = "ressources/testing/fake_LOC_val_solution.csv"
    imagenet_data = get_imagenet_data({'synset_path': synset_path,
                                       'train_dir': train_dir,
                                       'train_data_percentage': training_percentage,
                                       'val_dir': val_dir,
                                       'val_gt_path': val_gt_path})
    train_paths, train_labels, val_paths, val_labels = imagenet_data
    train_dataset = dataloader.get_dataset(train_paths, train_labels, n_classes=10, batch_size=2)

    model.fit(x=train_dataset,
              epochs=1,
              callbacks=[callback],
              max_queue_size=16,
              workers=8,
              )

    # check that ckpts were written
    files = os.listdir(ckpt_dir)
    self.assertTrue('checkpoint' in files)
    self.assertTrue('ckpt-1.index' in files)
    self.assertTrue('ckpt-1.data-00000-of-00002' in files)
    self.assertTrue('ckpt-1.data-00001-of-00002' in files)

    # train for 5 more epochs
    model.fit(x=train_dataset,
              epochs=5,
              callbacks=[callback],
              max_queue_size=16,
              workers=8,
              )

    # check that ckpt-1 was remove and ckpt 2,3,4,5,6 added
    files = os.listdir(ckpt_dir)
    self.assertTrue('checkpoint' in files)
    self.assertFalse('ckpt-1.index' in files)
    self.assertTrue('ckpt-2.index' in files)
    self.assertTrue('ckpt-3.index' in files)
    self.assertTrue('ckpt-4.index' in files)
    self.assertTrue('ckpt-5.index' in files)
    self.assertTrue('ckpt-6.index' in files)

    # check that we can load the last checkpoint
    del model
    del optimizer
    del callback
    model = Model1('tensorflow', factor=1).model
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    callback, latest_epoch = init_custom_checkpoint_callbacks(model, optimizer, ckpt_dir, max_ckpt=5, , save_frequency=1)
    self.assertEqual(latest_epoch, 6)

    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)
    shutil.rmtree(ckpt_dir)


def create_fake_training_data():
  dataset_dir = tempfile.mkdtemp()
  dirs = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331',
          'n01498041', 'n01514668', 'n01514859', 'n01518878']
  for d in dirs:
    os.makedirs(os.path.join(dataset_dir, d))
    for i in range(2):
      cv2.imwrite(os.path.join(dataset_dir, d, '{}_{}.JPEG'.format(d, i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
  return dataset_dir


def create_fake_val_data():
  dataset_dir = tempfile.mkdtemp()
  for i in range(2):
    cv2.imwrite(os.path.join(dataset_dir, 'ILSVRC2012_val_{}.JPEG'.format(i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
  return dataset_dir
