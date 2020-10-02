import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np
import yaml
from tfrecord_writer import build_tfrecord_dataset


class TestTfrecordWriter(unittest.TestCase):
  def test_process_images_in_class_directory(self):
    num_examples_each_class = 10
    data_dir = create_fake_dataset(num_examples_each_class)
    name = 'Test-dataset'
    description = 'A small test datset'
    tfrecord_dir_path = tempfile.mkdtemp()

    args = {'name': name, 'description': description, 'tfrecord_dir_path': tfrecord_dir_path,
            'tfrecord_size': 2, 'preprocessing': 'NO', 'image_size': (224, 224),
            'data': {'images_dir_path': data_dir,
                     'annotation_file_path': None,
                     'delimiter': ',',
                     'header_exists': False,
                     'split_names': ['train', 'validation', 'test'],
                     'split_percentages': [0.8, 0.1, 0.1],
                     }
            }
    build_tfrecord_dataset(args)

    dataset_info = load_yaml(data_dir=tfrecord_dir_path, dataset_name=name)

    # check newly created datset name and description
    self.assertEqual(name, dataset_info['name'])
    self.assertEqual(description, dataset_info['description'])

    i = 0
    # check split percentage
    for split_name, split_items in dataset_info['splits'].items():
        num_exmaples = split_items['num_examples']
        self.assertAlmostEqual(args['data']['split_percentages'][i], num_exmaples / (2.0 * num_examples_each_class))
        i += 1

    shutil.rmtree(data_dir)
    shutil.rmtree(tfrecord_dir_path)

  def test_process_with_annotation_file(self):
    num_examples_each_class = 10
    data_dir, annotation_file = create_fake_dataset_with_annotation_file(num_examples_each_class)
    name = 'Test-dataset'
    description = 'A small test datset'
    tfrecord_dir_path = tempfile.mkdtemp()

    args = {'name': name, 'description': description, 'tfrecord_dir_path': tfrecord_dir_path,
            'tfrecord_size': 2, 'preprocessing': 'NO', 'image_size': (224, 224),
            'data': {'images_dir_path': data_dir,
                     'annotation_file_path': annotation_file,
                     'delimiter': ',',
                     'header_exists': False,
                     'split_names': ['train', 'validation', 'test'],
                     'split_percentages': [0.8, 0.1, 0.1],
                     }
            }
    build_tfrecord_dataset(args)

    dataset_info = load_yaml(data_dir=tfrecord_dir_path, dataset_name=name)

    # check newly created datset name and description
    self.assertEqual(name, dataset_info['name'])
    self.assertEqual(description, dataset_info['description'])

    i = 0
    # check split percentage
    for split_name, split_items in dataset_info['splits'].items():
        num_exmaples = split_items['num_examples']
        self.assertAlmostEqual(args['data']['split_percentages'][i], num_exmaples / (2.0 * num_examples_each_class))
        i += 1

    shutil.rmtree(data_dir)
    shutil.rmtree(tfrecord_dir_path)


def load_yaml(data_dir, dataset_name):
  yaml_file = os.path.join(data_dir, dataset_name, 'dataset_info.yaml')
  with open(yaml_file, 'r') as stream:
    try:
      dataset_info = yaml.safe_load(stream)
    except yaml.YAMLError as e:
      print('Error parsing file', yaml_file)
      raise e
  return dataset_info


def create_fake_dataset(n_images_per_class=2):
  dataset_dir = tempfile.mkdtemp()
  os.makedirs(os.path.join(dataset_dir, 'cat'), exist_ok=True)
  os.makedirs(os.path.join(dataset_dir, 'dog'), exist_ok=True)
  for i in range(n_images_per_class):
    cv2.imwrite(os.path.join(dataset_dir, 'dog', '{}.jpg'.format(i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
    cv2.imwrite(os.path.join(dataset_dir, 'cat', '{}.jpg'.format(i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
  return dataset_dir


def create_fake_dataset_with_annotation_file(n_images_per_class=2):
  dataset_dir = tempfile.mkdtemp()
  os.makedirs(dataset_dir, exist_ok=True)

  annotation_file = os.path.join(dataset_dir, 'annotations.txt')

  labels = ['cat', 'dog']

  with open(annotation_file, 'w', encoding='utf-8') as f:
    for i in range(n_images_per_class*2):
      cv2.imwrite(os.path.join(dataset_dir, '{}.jpg'.format(i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
      line = '{}.jpg'.format(i) + "," + labels[i % 2] + "\n"
      f.write(line)

  return dataset_dir, annotation_file
