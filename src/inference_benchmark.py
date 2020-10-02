"""this script should be called by ../inference_benchmark.py in a docker
it has very few dependances so it should run on any environment having tensorflow installed
Please, only import official python package or tensorflow dependances
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
# from trt_convert import convert_to_tensorrt


def get_gpu_info() -> defaultdict:
  """
  Returns: e.g.
      defaultdict(<class 'list'>, 
      {
          'device': ['0', '1'], 
          'name': ['TITAN V', 'Quadro RTX 8000'], 
          'pci bus id': ['0000:41:00.0', '0000:07:00.0'], 
          'compute capability': ['7.0', '7.5']
      })
  """
  get_info = defaultdict(list)
  for i in tf.python.client.device_lib.list_local_devices():
    if "compute capability: " in i.physical_device_desc:
      for j in i.physical_device_desc.split(", "):
        get_info[j.split(": ")[0]].append(j.split(": ")[1])
  # TODO handle the else part
  return get_info


def parse_args():
  """Parse the command line and return the model name and the engine to benchmark

  Returns:
      tuple: name of the model, engine
  """
  desc = "Inference benchmark"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--model_name', type=str, default='VGG16', help='Specify the name of the model')
  # parser.add_argument('--model_path', type=str, default=None, help='Specify the model path') # TODO to be added later
  parser.add_argument('--export_tensorrt', default=False, type=lambda x: (str(x).lower() in ['true', 't', '1']), help='specify if model requires tensorrt conversion')
  parser.add_argument('--tensorrt_precision', type=str, default='FP32', help='Provide precision FP32 or FP16 for optimizing tensorrt')
  parser.add_argument('--engine', type=str, default="upstride_2", help='Specify the engine: tensorflow or upstride_X or mix_X')
  parser.add_argument('--profiler_path', type=str, default="/profiling/test", help='path where the tensorboard profiler will be saved')
  parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
  parser.add_argument('--factor', type=int, default=1, help='factor to divide the number of parameters in upstride')
  parser.add_argument('--n_layers_before_tf', type=int, default=0, help='when engine = mix_X, number of layers to write in upstride')
  parser.add_argument('--n_steps', type=int, default=20, help='number of iterations to benchmark speed')
  parser.add_argument('--xla', default=False, type=lambda x: (str(x).lower() in ['true', 't', '1']), help='number of iterations to benchmark speed')
  args = parser.parse_args()
  return args.model_name, args.export_tensorrt, args.tensorrt_precision, args.engine, args.profiler_path, args.batch_size, args.factor, args.n_layers_before_tf, args.n_steps, args.xla


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


def main():
  model_name, export_tensorrt, tensorrt_precision, engine, profiler_path, batch_size, factor, n_layers_before_tf, n_steps, xla = parse_args()

  # Solution come from https://stackoverflow.com/questions/48610132/tensorflow-crash-with-cudnn-status-alloc-failed
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)

  if xla:
    tf.config.optimizer.set_jit(True)
  tmp_dir = "/tmp/temp_dir"
  model = model_name_to_class[model_name](engine, factor, n_layers_before_tf=n_layers_before_tf).model
  n_params = model.count_params()
  tf.saved_model.save(model, tmp_dir)
  del model
  if export_tensorrt:
    # trt_path = convert_to_tensorrt(
    #     tmp_dir,
    #     image_size=[224, 224, 3],  # for now its hard coded.
    #     batch_size=batch_size,
    #     precision=tensorrt_precision
    # )
    # print(f'loading TensorRT model from path {trt_path}')
    # model = model_load_serve(trt_path)
    pass
  else:
    print(f'loading TF SaveModel from path {tmp_dir}')
    model = model_load_serve(tmp_dir)

  os.makedirs(profiler_path, exist_ok=True)

  # A few iteration to init the model
  print("first iteration")
  for data in random_data_iterator((batch_size, 224, 224, 3), 1, -1, 1):
    model(data)
  print("first iteration done")

  os.makedirs(profiler_path, exist_ok=True)
  tf.profiler.experimental.start(os.path.join(profiler_path, 'logs_{}'.format(engine)))
  for data in random_data_iterator((batch_size, 224, 224, 3), 5, -1, 1):
    model(data)
  tf.profiler.experimental.stop()

  # Benchmark
  start_time = time.time()
  for data in random_data_iterator((batch_size, 224, 224, 3), n_steps, -1, 1):
    model(data)
  end_time = time.time()

  try:
    gpu = get_gpu_info().get('name')[0]
  except TypeError:
    gpu = 'cpu only'

  output = {
      'total_time': end_time - start_time,
      'n_iterations': n_steps,
      'n_params': n_params,
      'tensorrt': export_tensorrt,
      'gpu': gpu
  }
  print(json.dumps(output))

  # clean up
  shutil.rmtree(tmp_dir)


if __name__ == "__main__":
  main()
