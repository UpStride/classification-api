import os
import tensorflow as tf
import upstride_argparse as argparse
from src.data import dataloader
from src.models import model_name_to_class
from src.models.generic_model import framework_list
from src.utils import check_folder, get_imagenet_data, model_dir
from submodules.global_dl import global_conf
from submodules.global_dl.training.training import create_env_directories, setup_mp, define_model_in_strategy, get_callbacks, init_custom_checkpoint_callbacks, GradientCallback
from submodules.global_dl.training import training
from submodules.global_dl.training import alchemy_api
from submodules.global_dl.training import export
from submodules.global_dl.training.optimizers import get_lr_scheduler, get_optimizer, arguments
from submodules.global_dl.training import optimizers
import numpy as np

tf.keras.backend.set_image_data_format('channels_first')
print(f"keras dataformat {tf.keras.backend.image_data_format()}")

arguments = [
    ['namespace', 'dataloader', dataloader.arguments],
    ['namespace', 'server', alchemy_api.arguments],
    ['namespace', 'optimizer', optimizers.arguments],
    ['namespace', 'export', export.arguments],
    [str, 'framework', 'tensorflow', 'Framework to use to define the model', lambda x: x in framework_list],
    [int, "factor", 1, 'division factor to scale the number of channel. factor=2 means the model will have half the number of channels compare to default implementation'],
    [int, 'n_layers_before_tf', 0, 'when using mix framework, number of layer defined using upstride', lambda x: x >= 0],
    [str, 'load_searched_arch', '', 'model definition file containing the searched architecture'],
    [str, "model_name", '', 'Specify the name of the model', lambda x: x in model_name_to_class],
    [float, "weight_decay", 0.0001, 'the weight decay value for l2 regularization', lambda x: x > 0.0], # TODO needs better organization
    ['namespace', 'conversion_params', [
        [bool, 'output_layer_before_up2tf', False, 'Whether to use final output layer before UpStride2TF conversion or not'],
        [str, 'tf2up_strategy', '', 'TF2UpStride conversion strategy'],
        [str, 'up2tf_strategy', 'default', 'UpStride2TF conversion strategy']
    ]],
] + global_conf.arguments + training.arguments


def main():
  """ function called when starting the code via command-line
  """
  args = argparse.parse_cmd(arguments)
  args['server'] = alchemy_api.start_training(args['server'])
  train(args)


def get_model(args):
  load_arch = args['load_searched_arch'] if args['load_searched_arch'] else None
  model = model_name_to_class[args['model_name']](args['framework'],
                                                  args['conversion_params'],
                                                  args['factor'],
                                                  args['input_size'],
                                                  args['num_classes'],
                                                  args['n_layers_before_tf'],
                                                  False,
                                                  weight_regularizer=tf.keras.regularizers.l2(args['weight_decay']),
                                                  load_searched_arch=load_arch).model

  model.summary()
  optimizer = get_optimizer(args['optimizer'])
  model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                metrics=['accuracy', 'top_k_categorical_accuracy'])
  # output the optimizer to save it in the checkpoint
  return model, optimizer

def get_experiment_name(args):
  experiment_dir = f"{args['model_name']}_{args['framework']}_factor{args['factor']}"
  if 'mix' in args['framework']:
    experiment_dir += "_mix_{}".format(args['n_layers_before_tf'])
  if args['configuration']['with_mixed_precision']:
    experiment_dir += "_mp"
  return experiment_dir

def get_dcn_dataloader():
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
  nb_classes                           = 10
  n_train                              = 45000
  shuf_inds  = np.arange(len(y_train))
  np.random.seed(0xDEADBEEF)
  np.random.shuffle(shuf_inds)
  train_inds = shuf_inds[:n_train]
  val_inds   = shuf_inds[n_train:]

  X_train    = X_train.astype('float32')/255.0
  X_test     = X_test .astype('float32')/255.0

  X_train_split = X_train[train_inds]
  X_val_split   = X_train[val_inds  ]
  y_train_split = y_train[train_inds]
  y_val_split   = y_train[val_inds  ]

  pixel_mean = np.mean(X_train_split, axis=0)

  X_train    = X_train_split.astype(np.float32) - pixel_mean
  X_val      = X_val_split  .astype(np.float32) - pixel_mean
  X_test     = X_test       .astype(np.float32) - pixel_mean

  Y_train    = tf.keras.utils.to_categorical(y_train_split, nb_classes)
  Y_val      = tf.keras.utils.to_categorical(y_val_split,   nb_classes)
  Y_test     = tf.keras.utils.to_categorical(y_test,        nb_classes)

  return X_train, Y_train, X_val, Y_val, X_test, y_test

def train(args):
  print(args)
  global_conf.config_tf2(args)
  checkpoint_dir, log_dir, export_dir = create_env_directories(args, get_experiment_name(args))

  # train_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['train_list'],
  #                                        num_classes=args["num_classes"], split=args['dataloader']['train_split_id'])
  # val_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['val_list'],
  #                                      num_classes=args["num_classes"], split=args['dataloader']['val_split_id'])
  # test_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['val_list'],
  #                                      num_classes=args["num_classes"], split='test')

  # dataloader specific to DCN source code
  X_train, Y_train, X_val, Y_val, X_test, y_test = get_dcn_dataloader()
  datagen         = tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range = 0.125,
                                      width_shift_range  = 0.125,
                                      horizontal_flip    = True)
  setup_mp(args)
  model, _ = define_model_in_strategy(args, get_model)
  tf.keras.utils.plot_model(model, to_file=os.path.join(args["log_dir"],"dcn_ours_model.png"),show_shapes=True)
  alchemy_api.send_model_info(model, args['server'])
  callbacks = get_callbacks(args, log_dir)

  # if args['debug']['log_gradients']:
  #   x, y = X_train[0], Y_train[0]
  #   gradient_cb = GradientCallback(batch=(x, y), log_dir=log_dir, log_freq=1)
  #   callbacks.append(gradient_cb)

  model_checkpoint_cb, latest_epoch = init_custom_checkpoint_callbacks({'model': model}, checkpoint_dir)
  callbacks.append(model_checkpoint_cb)
  if args['server']['id'] != '':
    callbacks.append(alchemy_api.send_metric_callbacks(args['server']))
  # model.fit(X_train,Y_train,
  #           validation_data=(X_val,Y_val),
  #           epochs=args['num_epochs'],
  #           callbacks=callbacks,
  #           max_queue_size=16,
  #           initial_epoch=latest_epoch
  #           )
  
  # specific to DCN source code.
  batch_size = args['dataloader']['batch_size']
  model.fit(datagen.flow(X_train, Y_train, batch_size),
                    steps_per_epoch = (len(X_train)+batch_size-1) // batch_size,
                    epochs          = args['num_epochs'],
                    verbose         = 1,
                    callbacks       = callbacks,
                    validation_data = (X_val, Y_val),
                    initial_epoch   = latest_epoch)
  print("Training Completed!!")
  model.evaluate((X_test, y_test),batch_size=batch_size)
  print("Evaluation Completed!!")
  # print("export model")
  # export.export(model, export_dir, args)

if __name__ == '__main__':
  main()
