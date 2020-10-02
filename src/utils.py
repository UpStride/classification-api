import csv
import glob
import os
import random
import cv2
import numpy as np
import tensorflow as tf
from shutil import copyfile


def copy_and_resize(source, dest, img_size):
  os.makedirs(dest, exist_ok=True)

  images_extensions = [".jpg", ".png", ".JPEG"]

  if source[-1] == "/":
    source = source[:-1]
  sources_len = len(source)
  for root, dirs, files in os.walk(source):
    for d in dirs:
      os.makedirs(os.path.join(dest, root[sources_len+1:], d), exist_ok=True)
    for f in files:
      if os.path.splitext(f)[1] in images_extensions:
        # then load, resize and save
        image = cv2.imread(os.path.join(root, f))
        image = cv2.resize(image, (img_size, img_size))
        r = cv2.imwrite(os.path.join(dest, root[sources_len+1:], f), image)
        if r == False:
          raise Exception("issue writing image {}".format(os.path.join(dest, root[sources_len+1:], f)))
      else:
        # simple copy
        copyfile(os.path.join(root, f), os.path.join(dest, root[sources_len+1:], f))


def model_dir(args):
  if args.model_name == 'resnet':
    return "{}{}".format(args.model_name, args.res_n)
  else:
    return "{}".format(args.model_name)


def get_synset(path: str):
  """Parse the LOC_synset_mapping.txt file given in imagenet dataset

  Args:
      path (str): path of the LOC_synset_mapping.txt file

  Returns:
      dict: dictionary mapping the label to the class id 
  """
  with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    synset_dict = {}
    for i, row in enumerate(csv_reader):
      synset_dict[row[0]] = i
  return synset_dict


def get_paths(dir: str):
  pattern = os.path.join(dir, '**', '*.JPEG')
  return glob.glob(pattern, recursive=True)


def get_partial_paths(dir, percentage):
  """

  :param dir: train data directory
  :param percentage: based on the percentage select the partial images for each class
  :return:
  """
  random.seed(1)
  percentage /= 100
  class_dirs = os.listdir(dir)
  paths = []
  for d in class_dirs:
    class_paths = glob.glob(os.path.join(dir, d, '*.JPEG'))
    random.shuffle(class_paths)
    end = int(len(class_paths) * percentage)
    paths += class_paths[0:end]
  return paths


def get_val_label_dict(val_gt_path: str):
  with open(val_gt_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader, None)  # skip the header
    val_dict = {}
    for row in csv_reader:
      val_dict[row[0]] = row[1].split(" ")[0]
  return val_dict


def get_imagenet_data(imagenet_data_args):
  """parse a imagenet dataset files and return usefull data for training and validation

  Args:
      synset_path (str): for instance "/home/user/upstride-tests/ILSVRC/LOC_synset_mapping.txt"
      train_dir (str): for instance "/home/user/upstride-tests/ILSVRC/Data/CLS-LOC/train/"
      training_percentage (int): 100 for training on the whole dataset
      val_dir (str): for instance "/home/user/upstride-tests/ILSVRC/Data/CLS-LOC/val/"
      val_gt_path (str): for instance "/home/user/upstride-tests/ILSVRC/LOC_val_solution.csv"

  Returns:
      tuple of 4 elements : train_paths, train_labels, val_paths, val_labels
      paths are lists of strings, labels are lists of integers
  """
  synset_path = imagenet_data_args['synset_path']
  train_dir = imagenet_data_args['train_dir']
  training_percentage = imagenet_data_args['train_data_percentage']
  val_dir = imagenet_data_args['val_dir']
  val_gt_path = imagenet_data_args['val_gt_path']

  synset = get_synset(synset_path)
  train_paths = get_paths(train_dir) if training_percentage == 100 else get_partial_paths(train_dir, training_percentage)
  train_labels = [synset[path.split("/")[-2]] for path in train_paths]

  # train data are shuffled
  random.seed(0)
  combined = list(zip(train_paths, train_labels))
  random.shuffle(combined)
  train_paths, train_labels = zip(*combined)

  val_label_dict = get_val_label_dict(val_gt_path)
  val_paths = get_paths(val_dir)
  val_labels = [synset[val_label_dict[path.split("/")[-1].split(".")[0]]] for path in val_paths]

  return train_paths, train_labels, val_paths, val_labels


def check_folder(log_dir):
  # TODO os.makedirs(..., exists_ok=True) does the job
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  return log_dir
