""" see https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin
"""

import os

import tensorflow as tf
import numpy as np
import upstride_argparse as argparse
from tensorboard.plugins import projector

from train import (arguments, create_env_directories, dataloader,
                   define_model_in_strategy, get_experiment_name, get_model,
                   init_custom_checkpoint_callbacks)


def find_from_upstride2tf():
  # this part is tricky. The idea is to go through the layers until the Upstride2TF operation and save pointers to interesting layers
  # then redefine the model with new outputs
  layer_before_conversion = None
  layer_after_conversion = None
  tf2upstride_seen = False
  for layer in model.layers:
    if tf2upstride_seen and layer_after_conversion is None:
      layer_after_conversion = layer
      break
    if type(layer) == UpStride2TF:
      tf2upstride_seen = True
      layer_before_conversion = previous_layer  # as UpStride2TF can't be the first layer, here previous_layer is defined
    previous_layer = layer
  return layer_before_conversion, layer_after_conversion


def find_tensor_by_keras_name(name, model):
  output = None
  for l in model.layers:
    if l.name == name:
      output = l
      break
  return output.output


def find_tensor_by_tf_name(name):
  return tf.python.keras.backend.get_graph().get_tensor_by_name(name)


def main():
  new_arguments = [
      [str, "keras_op", "", 'name of the op to select for feature visualization', lambda x: x != ""],
      [str, "embeddings_dir", "", "path to save the tensorboard visualization", lambda x: x != ""]
  ]
  args = argparse.parse_cmd(arguments + new_arguments)
  # define the model and load the checkpoint
  model, _ = define_model_in_strategy(args, get_model)
  checkpoint_dir, _, _ = create_env_directories(args, get_experiment_name(args))
  init_custom_checkpoint_callbacks({'model': model}, checkpoint_dir)
  # add the embeddings outputs and redefine the model
  output = find_tensor_by_keras_name(args['keras_op'], model)
  output = tf.keras.layers.GlobalAveragePooling2D()(output)
  emb_model = tf.keras.Model(inputs=model.layers[0].input, outputs=[output])
  # now prepare the dataset
  train_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['train_list'],
                                         num_classes=args["num_classes"], split=args['dataloader']['train_split_id'])
  # run the model on data to generate the embeddings
  for i, (x, y) in enumerate(train_dataset):
    if i == 10:  # todo add this parameter in argparse ?
      break
    emb = emb_model(x).numpy() if i == 0 else np.concatenate([emb, emb_model(x).numpy()])
    labels = tf.math.argmax(y, axis=1).numpy() if i == 0 else np.concatenate([labels, tf.math.argmax(y, axis=1).numpy()])
  # save the embeddings
  checkpoint = tf.train.Checkpoint(emb=tf.Variable(emb))  # , emb_after=emb_after)
  log_dir = args["embeddings_dir"]
  os.makedirs(log_dir, exist_ok=True)
  checkpoint.save(os.path.join(log_dir, "ckpt"))
  # set up config
  config = projector.ProjectorConfig()
  embedding = config.embeddings.add()
  embedding.tensor_name = "emb/.ATTRIBUTES/VARIABLE_VALUE"
  embedding.metadata_path = 'metadata.tsv'
  projector.visualize_embeddings(log_dir, config)
  # create metadata.tsv file
  with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    for label in labels:
      f.write(f"{label}\n")


if __name__ == "__main__":
  main()
