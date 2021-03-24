"""this script should be called by ../inference_benchmark.py in a docker
it has very few dependance so it should run on any environment having tensorflow installed
Please, only import official python package or tensorflow dependance
"""
import argparse
import json
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.saved_model import tag_constants

from models import model_name_to_class

import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../submodules/global_dl/training'))
from trt_convert import convert_to_tensorrt


# This method was working with previous version of Tensorflow, but doesn't seem to work with TF 2.4. 
# I didn't find anything in the doc to reimplement it with newer method of tensorflow. But maybe it will be 
# possible with next version of TF
# def get_gpu_info() -> defaultdict:
#   """
#   Returns: e.g.
#       defaultdict(<class 'list'>, 
#       {
#           'device': ['0', '1'], 
#           'name': ['TITAN V', 'Quadro RTX 8000'], 
#           'pci bus id': ['0000:41:00.0', '0000:07:00.0'], 
#           'compute capability': ['7.0', '7.5']
#       })
#   """
#   get_info = defaultdict(list)
#   for i in tf.config.list_physical_devices():
#     if "compute capability: " in i.physical_device_desc:
#       for j in i.physical_device_desc.split(", "):
#         get_info[j.split(": ")[0]].append(j.split(": ")[1])
#   # TODO handle the else part
#   return get_info


def str2bool(v):
  """custom argpase type to be able to parse boolean
  see : 
  https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
  """
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
  """Parse the command line and return the configuration to run the benchmark

  Returns:
      tuple: name of the model, engine
  """
  desc = "Inference benchmark"
  parser = argparse.ArgumentParser(description=desc)
  # Set of parameters needed for every benchmark
  parser.add_argument('--batch_size',    type=int,   default=1,            help='batch_size')
  parser.add_argument('--engine',        type=str,   default="tensorflow", help='Specify the engine: tensorflow or upstride_X')
  parser.add_argument('--factor',        type=float, default=1.,           help='factor to divide the number of channels')
  parser.add_argument('--model_name',    type=str,   default='VGG16',      help='Specify the name of the model')
  parser.add_argument('--n_steps',       type=int,   default=20,           help='number of iterations to benchmark speed')
  parser.add_argument('--profiler_path', type=str,   default="/tmp/prof",  help='path where the tensorboard profiler will be saved')
  parser.add_argument('--xla',           type=str2bool, nargs='?', const=True, default=False, help='if specify then run XLA compilation')

  # set or parameters specific to TensorRT
  parser.add_argument("--export_tensorrt",    type=str2bool, nargs='?', const=True, default=False, help="specify if model requires tensorrt conversion")
  parser.add_argument('--tensorrt_precision', type=str, default='FP32', help='Provide precision FP32 or FP16 for optimizing tensorrt')

  # If need to load a trained model
  parser.add_argument('--model_path',       type=str, default=None, help='Specify the model path')

  args = parser.parse_args()
  return vars(args)


def random_data_iterator(shape, n, min, max):
  """Simulate a dataset by generating random normalized images

  Args:
      shape (List): shape of the image. In most of the cases, (224, 224, 3)
      n (int): number of images to generate
      min (float): min of the random distribution
      max (float): max of the random distribution

  Yields:
      np.ndarray: generated images
  """
  for _ in range(n):
    data = np.random.random(shape)  # between [0, 1)
    data = data * (max-min) + min
    yield tf.constant(data.astype(np.float32))


def model_load_serve(path):
  saved_model = tf.saved_model.load(path, tags=[tag_constants.SERVING])
  model = saved_model.signatures['serving_default']
  return convert_to_constants.convert_variables_to_constants_v2(model)


def benchmark(config):
  # 1 Tensorflow configuration
  # GPU should be configured to have a progressive memory growth, else some specific configurations may crashed (TF 2.0 on RTX2000 for instance)
  physical_devices = tf.config.list_physical_devices('GPU')
  for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)
  if config['xla']:
    tf.config.optimizer.set_jit(True)

  # 2 Model preparation
  if config['engine'] == 'tensorflow':
    upstride_type = -1
  else:
    upstride_type = int(config['engine'][-1])

  kwargs = {
    'input_size': [224, 224, 3],
    'num_classes': 10,
    'factor': config['factor'],
    'upstride_type': upstride_type,
    'changing_ids': []
  }

  model = model_name_to_class[config['model_name']](**kwargs).build()
  n_params = model.count_params()

  # 3 Maybe convert to TensorRT.
  # To do this, we save the model, remove it from memory then reload it using TensorRT
  tmp_dir = "/tmp/temp_dir"
  if config['export_tensorrt']:
    # save the model
    tf.saved_model.save(model, tmp_dir)

    # Remove it 
    del model

    # Reload using tensorRT
    trt_path = convert_to_tensorrt(
        tmp_dir,
        image_size=[224, 224, 3],  # for now its hard coded.
        batch_size=config['batch_size'],
        precision=config['tensorrt_precision']
    )
    print(f'loading TensorRT model from path {trt_path}')
    model = model_load_serve(trt_path)

  # 4 prepare the environment
  os.makedirs(config['profiler_path'], exist_ok=True)

  # A few iteration to init the model
  print("first iteration")
  for data in random_data_iterator((config['batch_size'], 224, 224, 3), 1, -1, 1):
    model(data)
  print("first iteration done")

  os.makedirs(config['profiler_path'], exist_ok=True)
  tf.profiler.experimental.start(os.path.join(config['profiler_path'], 'logs_{}'.format(config['engine'])))
  for data in random_data_iterator((config['batch_size'], 224, 224, 3), 5, -1, 1):
    model(data)
  tf.profiler.experimental.stop()

  # Benchmark
  start_time = time.time()
  for data in random_data_iterator((config['batch_size'], 224, 224, 3), config['n_steps'], -1, 1):
    model(data)
  end_time = time.time()

  # TODO reactivate this part as soon as we know how to do it with modern version of TF
  # try:
  #   gpu = get_gpu_info().get('name')[0]
  # except TypeError:
  #   gpu = 'cpu only'

  output = {
      'total_time': end_time - start_time,
      'n_iterations': config['n_steps'],
      'n_params': n_params,
      'tensorrt': config['export_tensorrt'],
      # 'gpu': gpu
  }
  print(json.dumps(output))

  # clean up
  if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
  config = parse_args()
  benchmark(config)
