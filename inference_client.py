import zmq
import numpy as np
import tensorflow as tf
import upstride_argparse as argparse
from src.data import dataloader, augmentations

args_spec = [
    # dataloader specification to run inference on a public dataset
    [int, "num_classes", 0, 'Number of classes', lambda x: x > 0],
    ['namespace', 'dataloader', [
        ['list[str]', 'list', ['Resize', 'CentralCrop', 'Normalize'], 'Comma-separated list of data augmentation operations'],
        [str, "data_dir", '', "directory to read/write data. Defaults to  \"~/tensorflow_datasets\""],
        [str, 'name', None, 'Choose the dataset to be used', lambda x: not (x is None)],
        [str, 'split_id', 'validation', 'Split id in the dataset to use'],
        [int, 'batch_size', 1, 'The size of batch per gpu', lambda x: x > 0],
    ] + augmentations.arguments],

    # networking parameters
    [int, 'zmq_port', 5555, 'Specify the port to connect the ZMQ socket', lambda x: x > 0],
]


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
  total = val.shape[0]
  correct = [val[j][np.argmax(res[j])] == 1 for j in range(total)].count(True)
  return total, correct


def send_and_evaluate_dataset(dataset, socket):
  sent_records_count = 0
  logging_frequency = 10
  correct_count = 0
  images_count = 0
  for record in dataset:
    if sent_records_count % logging_frequency == 0:
      accuracy = 100.0 * correct_count / images_count if images_count > 0 else float("nan")
      print("Records sent: %d, accuracy: %0.2f%%" % (sent_records_count, accuracy))
    total, correct = send_and_evaluate_record(record, socket)
    images_count = images_count + total
    correct_count = correct_count + correct
    sent_records_count += 1

  print("Total records sent:", sent_records_count)
  return images_count, correct_count


def main():
  args = argparse.parse_cmd(args_spec)
  dataset = get_dataset(args)
  socket = create_zmq_socket(args['zmq_port'])
  images_count, correct_count = send_and_evaluate_dataset(dataset, socket)
  accuracy = correct_count / images_count
  print("Accuracy of the remote model:", accuracy)

if __name__ == '__main__':
  main()