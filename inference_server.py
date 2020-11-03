import os
import zmq
import numpy as np
import tensorflow as tf
import upstride_argparse as argparse
from src.data import dataloader, augmentations
from submodules.global_dl import global_conf

args_spec = [
    # framework specification
    [str, 'model_dir', None, 'Path to a folder containing saved model', lambda x: os.path.exists(x)],

    # dataloader specification to run inference on a dataset
    [int, "num_classes", 0, 'Number of classes'],
    ['namespace', 'dataloader', [
        ['list[str]', 'list', ['Resize', 'CentralCrop', 'Normalize'], 'Comma-separated list of data augmentation operations'],
        [str, "data_dir", '', "directory to read/write data. Defaults to  \"~/tensorflow_datasets\""],
        [str, 'name', None, 'Choose the dataset to be used'],
        [str, 'split_id', 'validation', 'Split id in the dataset to use'],
        [int, 'batch_size', 1, 'The size of batch per gpu', lambda x: x > 0],
    ] + augmentations.arguments],

    # networking parameters
    [int, 'zmq_port', 5555, 'Specify the port to connect the ZMQ socket', lambda x: x > 0],
] + global_conf.arguments



def load_model(args):
  from train import get_experiment_name
  # import upstride to enable model deserialization
  import upstride.type0.tf.keras.layers
  import upstride.type2.tf.keras.layers
  print("Loading model from", args['model_dir'])
  model = tf.keras.models.load_model(args['model_dir'])
  return model


def evaluate_dataset(args, model):
    print(f"Evaluating on {args['dataloader']['name']}")
    args['dataloader']['train_split_id'] = None
    dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['val_list'],
                                     num_classes=args["num_classes"], split=args['dataloader']['val_split_id'])
    model.evaluate(dataset)


def create_zmq_socket(zmq_port):
  context = zmq.Context()
  socket = context.socket(zmq.REP)
  socket.bind("tcp://*:" + str(zmq_port))
  return socket


def process_incoming_image_batches(model, shape, socket):
  received_messages_count = 0
  logging_frequency = 1000

  # set batch dimension to -1 for reshaping
  if shape[0] is None:
      shape[0] = -1

  # loop forever processing incoming messages
  while True:
    if received_messages_count % logging_frequency == 0:
      print(f"Received {received_messages_count} messages")

    message = socket.recv()
    img = np.frombuffer(message, dtype='float32').reshape(shape)
    res = model.predict(img)
    socket.send(res)
    
    received_messages_count += 1


def main():
  """ CLI entry point
  """
  # parse arguments
  args = argparse.parse_cmd(args_spec)

  # perform global configuration (XLA and memory growth)
  global_conf.config_tf2(args)

  # load model
  model = load_model(args)
  model.summary()

  # if dataloader.name is set, evaluating on a specific dataset
  if args['dataloader']['name'] is not None:
    evaluate_dataset(args, model)

  # otherwise for images listen to a zmq socket
  else:
    socket = create_zmq_socket(args['zmq_port'])
    assert len(model.inputs) == 1, "Cannot find model input to send images on"
    process_incoming_image_batches(model, list(model.inputs[0].shape), socket)


if __name__ == '__main__':
  main()
