import os
import sys
import shutil
import tempfile
import unittest
import cv2
import numpy as np
from src.data.dataloader import TFRecordExtractor

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'scripts'))
from tfrecord_by_separate_dir_or_annotation_file import build_tfrecord_dataset


class TestTfrecordExtractor(unittest.TestCase):
    def test_process(self):
        tfrecord_dir_path, dataset_name = create_dataset()

        # Retrieve the train tf records
        train_dataset_extractor = TFRecordExtractor(dataset_name, tfrecord_dir_path, "train")
        for image, label in train_dataset_extractor.get_tf_dataset().take(1):
            # Check train image shape
            self.assertEqual(image.shape, (640, 480, 3))

        val_dataset_extractor = TFRecordExtractor(dataset_name, tfrecord_dir_path, "validation")
        for image, label in val_dataset_extractor.get_tf_dataset().take(1):
            # Check validation image shape
            self.assertEqual(image.shape, (520, 380, 3))

        test_dataset_extractor = TFRecordExtractor(dataset_name, tfrecord_dir_path, "test")
        for image, label in test_dataset_extractor.get_tf_dataset().take(1):
            # Check test image shape
            self.assertEqual(image.shape, (520, 380, 3))

        shutil.rmtree(tfrecord_dir_path)


def create_dataset():
    TRAIN_EXAMPLE_PER_CLASS = 10
    VAL_EXAMPLE_PER_CLASS = 5
    TEST_EXAMPLE_PER_CLASS = 4
    train_dir = create_fake_dataset_from_directory(TRAIN_EXAMPLE_PER_CLASS)
    val_dir, val_annotation_file = create_fake_dataset_with_annotation_file(VAL_EXAMPLE_PER_CLASS)
    test_dir, test_annotation_file = create_fake_dataset_with_annotation_file(TEST_EXAMPLE_PER_CLASS)
    name = 'Test-dataset'
    description = 'A small test datset'
    tfrecord_dir_path = tempfile.mkdtemp()

    args = {'name': name, 'description': description, 'tfrecord_dir_path': tfrecord_dir_path,
            'tfrecord_size': 2, 'preprocessing': 'NO', 'image_size': (224, 224),
            'train': {'images_dir_path': train_dir,
                      'annotation_file_path': None,
                      'delimiter': ',',
                      'header_exists': False,
                      },
            'validation': {'images_dir_path': val_dir,
                           'annotation_file_path': val_annotation_file,
                           'delimiter': ',',
                           'header_exists': False,
                           },
            'test': {'images_dir_path': test_dir,
                     'annotation_file_path': test_annotation_file,
                     'delimiter': ',',
                     'header_exists': False,
                     }
            }
    build_tfrecord_dataset(args)

    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)
    shutil.rmtree(test_dir)

    return tfrecord_dir_path, name


def create_fake_dataset_from_directory(n_images_per_class=2):
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
      cv2.imwrite(os.path.join(dataset_dir, '{}.jpg'.format(i)), np.ones((520, 380, 3), dtype=np.uint8) * 255)
      line = '{}.jpg'.format(i) + "," + labels[i % 2] + "\n"
      f.write(line)

  return dataset_dir, annotation_file
