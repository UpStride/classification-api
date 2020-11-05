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


def train(args):
  print(args)
  global_conf.config_tf2(args)
  checkpoint_dir, log_dir, export_dir = create_env_directories(args, get_experiment_name(args))

  train_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['train_list'],
                                         num_classes=args["num_classes"], split=args['dataloader']['train_split_id'])
  val_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['val_list'],
                                       num_classes=args["num_classes"], split=args['dataloader']['val_split_id'])

  setup_mp(args)
  model, _ = define_model_in_strategy(args, get_model)
  alchemy_api.send_model_info(model, args['server'])
  callbacks = get_callbacks(args, log_dir)

  if args['debug']['log_gradients']:
    x, y = next(iter(train_dataset.take(1)))
    gradient_cb = GradientCallback(batch=(x, y), log_dir=log_dir, log_freq=1)
    callbacks.append(gradient_cb)

  model_checkpoint_cb, latest_epoch = init_custom_checkpoint_callbacks({'model': model}, checkpoint_dir)
  callbacks.append(model_checkpoint_cb)
  if args['server']['id'] != '':
    callbacks.append(alchemy_api.send_metric_callbacks(args['server']))
  model.fit(x=train_dataset,
            validation_data=val_dataset,
            epochs=args['num_epochs'],
            callbacks=callbacks,
            max_queue_size=16,
            initial_epoch=latest_epoch
            )
  print("export model")
  export.export(model, export_dir, args)
  print("Training Completed!!")


if __name__ == '__main__':
  main()
