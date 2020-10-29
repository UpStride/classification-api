import os
import zmq
import numpy as np
import tensorflow as tf
import upstride_argparse as argparse
from src.data import dataloader, augmentations
from src.models.generic_model import framework_list
from src.models import model_name_to_class
from submodules.global_dl import global_conf
from train import arguments as args_spec

args_spec.append(
    [int, 'zmq_port', 5555, 'Specify the port to connect the ZMQ socket', lambda x: x > 0]
)


def load_model(args):
  from train import get_experiment_name
  # restoring from a saved model
  if args['export']['dir']:
    path = os.path.join(args['export']['dir'], get_experiment_name(args))
    # import upstride to enable model deserialization
    if args['framework'].startswith('upstride'):
      import upstride.type0.tf.keras.layers
      import upstride.type2.tf.keras.layers
    model = tf.keras.models.load_model(path)
  # restoring from a checkpoint
  elif args['checkpoint_dir']:
    ckpt_dir = os.path.join(args['checkpoint_dir'], get_experiment_name(args))
    model = model_name_to_class[args['model_name']](args['framework'],
                                                    args['factor'],
                                                    args['input_size'],
                                                    args['num_classes'],
                                                    args['n_layers_before_tf'], False).model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir, max_to_keep=None)
    restored_ckpt = manager.restore_or_initialize()
    if restored_ckpt is None:
      raise RuntimeError(f"Cannot restore from a checkpoint in {ckpt_dir}")
    print(f'Restoring {manager.latest_checkpoint}')
  else:
    raise ValueError('No trained model location is given. Specify checkpoint_dir or export.dir.')
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

  while True:
    if received_messages_count % logging_frequency == 0:
      print("received_messages_count: " + str(received_messages_count))

    message = socket.recv()
    img = np.frombuffer(message, dtype='float32')
    img = img.reshape(np.concatenate(([-1], shape)))
    res = model.predict(img)
    socket.send(res)
    
    received_messages_count = received_messages_count + 1


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

  else:
    socket = create_zmq_socket(args['zmq_port'])
    process_incoming_image_batches(model, args['input_size'], socket)


if __name__ == '__main__':
  main()
