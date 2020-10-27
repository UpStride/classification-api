import zmq
import numpy as np
import tensorflow as tf
import upstride_argparse as argparse
from src.data import dataloader, augmentations
from src.models.generic_model import framework_list
from src.models import model_name_to_class
from submodules.global_dl import global_conf

args_spec = [
    # model specification
    [int, "num_classes", 0, 'Number of classes', lambda x: x > 0],

    # dataloader specification to run inference on a public dataset
    ['namespace', 'dataloader', [
        ['list[str]', 'list', ['Resize', 'CentralCrop', 'Normalize'], 'Comma-separed list of data augmentation operations'],
        [str, "data_dir", '', "directory to read/write data. Defaults to  \"~/tensorflow_datasets\""],
        [str, 'name', None, 'Choose the dataset to be used for training'],
        [str, 'split_id', 'validation', ''],
        [int, 'batch_size', 1, 'The size of batch per gpu', lambda x: x > 0],
    ] + augmentations.arguments],

    # other stuff to resolve the experiment name
    ['namespace', 'configuration', [[bool, "with_mixed_precision", False, 'To train with mixed precision']]],

    [int, 'zmq_port', 5555, 'Specify the port to connect the ZMQ socket', lambda x: x > 0],
] + global_conf.arguments


def get_dataset(args):
  args['dataloader']['train_split_id'] = None
  dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['list'],
                                     num_classes=args["num_classes"], split=args['dataloader']['split_id'])
  return dataset


def create_zmq_socket(port):
  context = zmq.Context()
  socket = context.socket(zmq.REQ)
  socket.connect("tcp://localhost:" + str(port))
  return socket


def send_and_evaluate_record(record, socket):
  img = record[0].numpy()
  val = record[1].numpy()

  socket.send(img)
  reply = socket.recv()

  res = np.frombuffer(reply, dtype='float32').reshape(val.shape)

  images = val.shape[0]
  correct = [val[j][np.argmax(res[j])] == 1 for j in range(images)].count(True)
  
  return images, correct


def send_and_evaluate_dataset(dataset, socket):
  sent_records_count = 0
  logging_frequency = 1000
  correct_count = 0
  images_count = 0
  for record in dataset:
    if sent_records_count % logging_frequency == 0:
      print("sent_records_count: " + str(sent_records_count))

    images, correct = send_and_evaluate_record(record, socket)
    images_count = images_count + images
    correct_count = correct_count + correct

    sent_records_count = sent_records_count + 1

  print("Total records sent: " + str(sent_records_count))
  return images_count, correct_count

def main():

  # parse arguments
  args = argparse.parse_cmd(args_spec)

  # perform global configuration (XLA and memory growth)
  global_conf.config_tf2(args)

  if args['dataloader']['name'] is not None:
    dataset = get_dataset(args)
    socket = create_zmq_socket(args['zmq_port'])
    images_count, correct_count = send_and_evaluate_dataset(dataset, socket)
    accuracy = correct_count / images_count
    print("Accuracy of the remote model: " + str(accuracy))

if __name__ == '__main__':
  main()