"""
This script prepares a classification dataset for efficient tensorflow training by creating train/validation/test split
based on their split percentages of the whole data

It assumes the images are store in a directory with one subdirectory per classes or an annotation file provided where
path in the format `1st column`: images names, `2nd column`: label. if it is `None` it will be assumed that images are
stored in the sub directories of `annotation_file_path`

"""

import glob
import logging
import os
import random
from typing import List

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from submodules.global_dl.tfrecord import create_tfrecord
from submodules.global_dl.tfrecord.create_tfrecord import IMAGE_EXTENSIONS, PREPROCESSING_STRATEGIES
from upstride_argparse import parse_cmd


class UpStrideDatasetBuilder:
    def __init__(self, name: str, description: str, tfrecord_dir_path: str, class_names: List[str], splits: List[create_tfrecord.Split],
                 tfrecord_size=10000, preprocessing="NO", image_size=(224, 224)):
        """ Dataset builder class

            Args:
                name: name of the dataset
                description: description of the dataset
                class_names: class names of the dataset
                tfrecord_dir_path: directory path to write the tfrecords
                tfrecord_size: number of images to write in a tfrecord file
                preprocessing: one of PREPROCESSING_STRATEGIES
                image_size: (224, 224) the size of the image to load in the tfrecord
        """
        self.metadata = {"name": name, "description": description, 'splits': {}, "class_names": class_names}
        self.tfrecord_dir_path = os.path.join(tfrecord_dir_path, name)
        self.splits = splits
        self.tfrecord_size = tfrecord_size
        self.preprocessing = preprocessing
        assert len(image_size) == 2, f"image_size must have a len of 2"
        self.image_size = image_size

    def __tfrecord_files_list(self, split_name):
        """
        list all the tfrecords file in the tfrecord_dir_path for a particular dataset split
        Args
            split_name: split name
        """
        return [os.path.basename(path) for path in
                glob.glob(os.path.join(self.tfrecord_dir_path, split_name + "*.tfrecord"))]

    def __store_tfrecord(self, split_name, path_label_list):
        """
        Store the tfrecord for a specific split in the tf record directory

        """
        print(f"Creating {split_name} split.....")
        store_images_in_tfrecords(path_label_list, self.tfrecord_dir_path, split_name, self.tfrecord_size,
                                  self.preprocessing, self.image_size)

    def __store_dataset_metadata(self):
        """
        Store the dataset metadata in the tf record directory

        """
        meta_file_path = os.path.join(self.tfrecord_dir_path, 'dataset_info.yaml')
        # check if the file already exists:
        if os.path.exists(meta_file_path):
            with open(meta_file_path, 'r') as stream:
                try:
                    dataset_info = yaml.safe_load(stream)
                except yaml.YAMLError as e:
                    print('Error parsing file', meta_file_path)
                    raise e

            if dataset_info["name"] == self.metadata["name"]:
                for split_name, items in dataset_info["splits"].items():
                    self.metadata['splits'][split_name] = items

        with open(os.path.join(meta_file_path), 'w') as f:
            yaml.dump(self.metadata, f, default_flow_style=False, sort_keys=False)

    def build(self):
        """
        Store the tfrecords, then add metadata in self.metadata and finally store dataset metadata
        """
        for split in self.splits:
            self.__store_tfrecord(split.name, split.path_label_list)
            self.metadata['splits'][split.name] = {"tfrecord_files": self.__tfrecord_files_list(split.name),
                                                   "num_examples": len(split.path_label_list)}
        self.__store_dataset_metadata()


def load_and_preprocess_img(img_path: str, final_shape, preprocessing: str):
    """
    Load an image, process it and convert it to a friendly tensorflow format
    Args:
        img_path: path of the image to load
        final_shape: Shape that the image will have after the preprocessing. For instance (224, 224)
        preprocessing: preprocessing to apply one among ["NO", "CENTER_CROP_THEN_SCALE", "SQUARE_MARGIN_THEN_SCALE"]

    Returns:

    """
    if preprocessing not in PREPROCESSING_STRATEGIES:
        logging.error(f"{preprocessing} is not in list {PREPROCESSING_STRATEGIES}")
        raise TypeError()

    img = cv2.imread(img_path)
    shape = img.shape
    if preprocessing == "CENTER_CROP_THEN_SCALE":
        # Center-crop the image
        min_shape = min(shape[0], shape[1])
        crop_beginning = [int((shape[0] - min_shape) / 2),
                          int((shape[1] - min_shape) / 2)]
        img = img[crop_beginning[0]:crop_beginning[0] + min_shape,
              crop_beginning[1]:crop_beginning[1] + min_shape, :]
    elif preprocessing == "SQUARE_MARGIN_THEN_SCALE":
        max_shape = max(shape[0], shape[1])
        new_image = np.zeros((max_shape, max_shape, 3), dtype=np.uint8)
        upper_left_point = [int((max_shape - shape[i]) / 2) for i in range(2)]

        a = upper_left_point[0]
        b = upper_left_point[1]
        new_image[a: a + shape[0], b: b + shape[1], :] = img
        img = new_image
    # Resize the image
    if preprocessing != "NO":
        img = cv2.resize(img, final_shape)
        shape = final_shape
    img_str = cv2.imencode('.jpg', img)[1].tobytes()
    return img_str, shape[0], shape[1]


def store_images_in_tfrecords(path_label_list: List[dict], tfrecord_dir_path: str, tfrecord_prefix: str,
                              tfrecord_size: int,
                              preprocessing="NO", image_size=(224, 224)):
    """
        store images in tf records
    Args:
        path_label_list: list of image info, each contain path and class id
        tfrecord_dir_path: directory path to write the tfrecords
        tfrecord_prefix: dataset split name
        tfrecord_size: number of images to write in a tfrecord file
        preprocessing:  one of PREPROCESSING_STRATEGIES
        image_size: size of the image to load in the tfrecord
    """

    tfrecord_manager = create_tfrecord.TfRecordManager(tfrecord_dir_path, tfrecord_prefix, tfrecord_size)
    # Store all images in tfrecord
    for i, image_info in enumerate(path_label_list):
        try:
            img, height, width = load_and_preprocess_img(image_info['path'], image_size, preprocessing)
            feature = {'image': create_tfrecord.bytes_feature(img), 'label': create_tfrecord.int64_feature(image_info['id']),
                       'size': create_tfrecord.int64_list_feature([height, width])}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            tfrecord_manager.add(example)
        except Exception as e:
            logging.warning('error with image', image_info['path'], ':', e)

    print("Done")


def group_image_paths_by_classes_from_dir(data_dir):
    """
    if images are stored in the sub-directories of data_dir by the name of their classes, the we can get the class names
    data_dir
        class1
            img1.jpg
            img2.jpg
        class2
            img1.jpg
            img2.jpg
        class3
            img1.jpg
            img2.jpg
    and create a group by class dictionary in the format {class1: [img1.jpg, img2.jpg], class3: [img1.jpg, img2.jpg, img3.jpg]}
    from the annotation file

    """

    if not os.path.isdir(data_dir):
        raise ValueError(f"There is no such directory by {data_dir}")

    # Find all the dirs containing the images
    classes_names = os.listdir(data_dir)
    classes_names = list(filter(lambda x: os.path.isdir(os.path.join(data_dir, x)), classes_names))
    classes_names.sort()

    class_map = {}
    for class_name in classes_names:
        # images are stored in the sub-directories named by their class names
        for root, dirs, files in os.walk(os.path.join(data_dir, class_name)):
            for f in files:
                if f.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                    if class_name not in class_map:
                        class_map[class_name] = [os.path.join(data_dir, root, f)]
                    else:
                        class_map[class_name].append(os.path.join(data_dir, root, f))

    return class_map, classes_names


def group_image_paths_by_classes_from_annotation_file(data_dir: str, annotation_file_path: str, header_exists=False, delimiter=','):
    """
    Create a group by class dictionary in the format {class1: [img1.jpg, img2.jpg], class3: [img1.jpg, img2.jpg, img3.jpg]}
    from the annotation file

    Args:
        data_dir: directory containing the images, will help to determine absolute paths of the images
        annotation_file_path: Annotation file path, 1st column:image names and 2nd column:class names
        header_exists: whether there is a header or not
        delimiter:  Delimiter which splits the columns
    return:
        annotation_map: absolute image paths are grouped in a dictionary by their class names
        classes_names: list of class names
    """

    if not os.path.exists(annotation_file_path):
        raise FileNotFoundError(f"Provided annotation file {annotation_file_path} does not exist")

    skip_rows = 1 if header_exists else 0
    pd_df = pd.read_csv(annotation_file_path, sep=delimiter, header=None, index_col=0, skiprows=skip_rows).iloc[:, 0]
    classes_names = pd_df.unique().tolist()
    classes_names.sort()

    annotation_map = pd_df.to_dict()

    class_map = {}
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            # Look for only the image files
            if f.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                class_name = annotation_map[f]
                if class_name not in class_map:
                    class_map[class_name] = [os.path.join(data_dir, root, f)]
                else:
                    class_map[class_name].append(os.path.join(data_dir, root, f))

    return class_map, classes_names


def build_tfrecord_dataset(args):
  if args["tfrecord_dir_path"] == '':
    raise ValueError("Directory for storing tf records is not provided!")
  if args["name"] == '':
    raise ValueError("Dataset name is not provided!")
  if args["data"]["images_dir_path"] == '':
    raise ValueError("Image Dataset directory is not provided!")
  
  print(f"Building tfrecord for dataset named {args['name']} .....")

  split_names = args['data']['split_names']
  split_percentages = args['data']['split_percentages']

  if len(split_names) != len(split_percentages):
      raise ValueError("split_names and split_percentages lengths should  be  same")

  total_percentage = 0
  for i in range(len(split_names)):
    total_percentage += split_percentages[i]
    if not (0. <= split_percentages[i] <= 1.):
      raise ValueError(
          f"{split_names[i]} split percentage should in range 0~1, but given "
          f"{split_percentages[i]}")

  if not total_percentage == 1:
    raise ValueError(f"Total percentage of all splits should equal to 1, but got {total_percentage}")

  if args['data']["annotation_file_path"]:
    # annotation maps are stored in a file
    class_map, classes_names = group_image_paths_by_classes_from_annotation_file(args["data"]["images_dir_path"],
                                                                                 args['data']["annotation_file_path"],
                                                                                 args['data']["header_exists"],
                                                                                 args['data']["delimiter"])
  else:
    # images are stored in the sub-directories named by their class names
    class_map, classes_names = group_image_paths_by_classes_from_dir(args["data"]["images_dir_path"])

  split_dict = {split_name: [] for split_name in split_names}
  # Now look over all the images of all the classes and split them
  for i, class_name in enumerate(classes_names):
    images = [{'path': path, 'id': i} for path in class_map[class_name]]
    random.shuffle(images)

    # splitting the images list based on the percentages
    start = 0
    for j in range(0, len(split_names)-1):
      num_samples = int(split_percentages[j]*len(images))
      split_dict[split_names[j]] += images[start:(start+num_samples)]
      start += num_samples
    split_dict[split_names[-1]] += images[start:]

  # shuffling the splits and creating Split objects
  splits = []
  for split_name, split_items in split_dict.items():
    if split_items:
      random.shuffle(split_items)
      splits.append(create_tfrecord.Split(split_name, split_items))

  n_classes = len(classes_names)
  print(f"Found {n_classes} classes")
  builder = UpStrideDatasetBuilder(args["name"], args["description"], args["tfrecord_dir_path"], classes_names,
                                   splits, args["tfrecord_size"], args["preprocessing"], tuple(args["image_size"]))
  builder.build()

  print(f"Dataset creation complete and stored in {args['tfrecord_dir_path']}")


dataset_arguments = [
    [str, 'name', "", 'Name of the dataset'],
    [str, 'description', "", 'Description of the dataset'],
    [str, 'tfrecord_dir_path', "", 'Directory where to store tfrecords'],
    [int, 'tfrecord_size', 10000, 'Number of images to be stored for each file'],
    [str, 'preprocessing', "NO", 'preprocessing: preprocessing to apply one among ["NO", "CENTER_CROP_THEN_SCALE", "SQUARE_MARGIN_THEN_SCALE"]'],
    ['list[int]', 'image_size', (224, 224), 'Shape that the image will have after the preprocessing. For instance (224, 224)'],
    ['namespace', 'data', [
        [str, 'images_dir_path', '', 'directory path for the images'],
        [str, 'annotation_file_path', '', 'annotation file path in the format `1st column`: images names, `2nd column`: label. '
         'if it is `None` it will be assumed that images are stored in the sub directories of `images_dir_path`'
         'by the name of their classes'],
        [str, 'delimiter', ',', 'Delimiter to split the annotation file columns'],
        [bool, 'header_exists', False, 'whether there is any header in the annotation file'],
        ['list[str]', 'split_names', ['train', 'validation', 'test'], 'Names of the splits'],
        ['list[float]', 'split_percentages', [0.8, 0.1, 0.1], 'Percentages of the splits'],
    ]]
]

if __name__ == "__main__":
  args = parse_cmd(dataset_arguments)
  build_tfrecord_dataset(args)
