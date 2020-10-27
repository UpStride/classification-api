import zmq
import imagezmq
import base64
import socket
import numpy as np
import tensorflow as tf
import upstride_argparse as argparse
from keras.preprocessing import image
from src.data import dataloader, augmentations
from src.models.generic_model import framework_list
from src.models import model_name_to_class
from submodules.global_dl import global_conf
import matplotlib.pyplot as plt

args_spec = [
    # ['list[str]', "yaml_config", [], "config file overriden by these argparser parameters"],

    # framework specification
    # [str, 'framework', 'tensorflow', 'Framework to use to define the model', lambda x: x in framework_list],

    # model specification
    # [str, "model_name", None, 'Specify the name of the model', lambda x: x in model_name_to_class],
    # [int, "factor", 1, 'division factor to scale the number of channel. factor=2 means the model will have half the number of channels compare to default implementation'],
    # [int, 'n_layers_before_tf', 0, 'when using mix framework, number of layer defined using upstride', lambda x: x >= 0],
    [int, "num_classes", 0, 'Number of classes', lambda x: x > 0],
    # ['list[int]', "input_size", [224, 224, 3], 'processed shape of each image'],

    # checkpoints directory
    # [str, "checkpoint_dir", None, 'Checkpoints directory to load the trained model from'],

    # dataloader specification to run inference on a public dataset
    ['namespace', 'dataloader', [
        ['list[str]', 'list', ['Resize', 'CentralCrop', 'Normalize'], 'Comma-separed list of data augmentation operations'],
        [str, "data_dir", '', "directory to read/write data. Defaults to  \"~/tensorflow_datasets\""],
        [str, 'name', None, 'Choose the dataset to be used for training'],
        [str, 'split_id', 'validation', ''],
        [int, 'batch_size', 1, 'The size of batch per gpu', lambda x: x > 0],
    ] + augmentations.arguments],

    # other stuff to resolve the experiment name
    # ['namespace', 'configuration', [[bool, "with_mixed_precision", False, 'To train with mixed precision']]],

    [int, 'zmq_port', 5555, 'Specify the port to connect the ZMQ socket', lambda x: x > 0],
] + global_conf.arguments


def get_dataset(args):
  args['dataloader']['train_split_id'] = None
  dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['list'],
                                     num_classes=args["num_classes"], split=args['dataloader']['split_id'])
  return dataset


def create_socket(port):
  context = zmq.Context()

  socket = context.socket(zmq.REQ)
  socket.connect("tcp://localhost:" + str(port))
  return socket


def send_dataset(dataset, args, use_imagezmq = False):
  if use_imagezmq:
    sender = imagezmq.ImageSender(connect_to='tcp://localhost:' + str(args['zmq_port']))
    sender_name = 'Albert' # socket.gethostname()

    i = 0

    for img in dataset:
      i = i + 1

      print(img[0])

      reply = sender.send_image(sender_name, img[0].numpy())
      res = np.frombuffer(reply, dtype='float32')
      print(res)
      break

      if i == 10:
        break

  else:
    socket = create_socket(args['zmq_port'])

    i = 0
    for img in dataset:
      i = i + 1

      socket.send(img[0].numpy())
      reply = socket.recv()

      res = np.frombuffer(reply, dtype='float32')
      print(res)
      break

      if i == 10:
        break


def main():

  # parse arguments
  args = argparse.parse_cmd(args_spec)

  # perform global configuration (XLA and memory growth)
  global_conf.config_tf2(args)

  if args['dataloader']['name'] is not None:
    # evaluate_dataset(args, model)

    dataset = get_dataset(args)
    send_dataset(dataset, args)


if __name__ == '__main__':
  main()