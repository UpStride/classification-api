from src.models.pdart import OPS as tf_ops
import torch.nn as nn
import torch
import tensorflow as tf
import numpy as np
import logging
import coloredlogs

coloredlogs.install(level='DEBUG')

import sys
sys.path.append('../pdarts')
from operations import OPS as pt_ops


# import official P-Dart implementation


def compare_no_kernel(tf_layer, pt_layer, name):
  """ compare two layers that don't need a kernel
  """
  # TODO test several values of size, C and stride
  
  for stride in [1, 2]:
    for c in [3, 6, 18]:
      for img_size in [3, 4, 9, 18, 224]:
        logging.debug(f"op {name}  stride {stride}, channel {c}, img_size {img_size}")
        inputs = np.random.normal(size=(1, c, img_size, img_size)).astype(np.float32)
        if stride == 2 and name == "skip_connect":
          continue
        pt_output = pt_layer(c, stride, False)(torch.tensor(inputs)).numpy()
        tf_output = tf_layer(tf.keras.layers, c, stride, False)(inputs).numpy()
        if not np.allclose(pt_output, tf_output, atol=1.e-6):
          breakpoint()

def compare_kernel(tf_layer, pt_layer):
  """ Compare two layers what need synchronization between there kernels
  """
  # Define the TF layer

  # get its kernel

  # apply its kernel to the pytorch layer

  # Test the two layers


def main():
  # New python engine are only channel first (c++ and python)
  tf.keras.backend.set_image_data_format('channels_first')

  no_kernel_ops = ['none',
                   'avg_pool_3x3',
                   'max_pool_3x3',
                   'skip_connect']
  kernel_ops = ['sep_conv_3x3',
                'sep_conv_5x5',
                'sep_conv_7x7',
                'dil_conv_3x3',
                'dil_conv_5x5',
                'conv_7x1_1x7']

  for key in no_kernel_ops:
    logging.debug(f"test {key}")
    pt_layer = pt_ops[key]
    tf_layer = tf_ops[key]
    compare_no_kernel(tf_layer, pt_layer, key)


if __name__ == "__main__":
  main()
