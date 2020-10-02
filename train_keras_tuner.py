import os
import tensorflow as tf
import upstride_argparse as argparse
from kerastuner.tuners import Hyperband, BayesianOptimization
from src.data import dataloader
from src.models import model_name_to_class
from src.models.generic_model import framework_list
from src.utils import check_folder, get_imagenet_data, model_dir
from submodules.global_dl import global_conf
from submodules.global_dl.training.training import create_env_directories, setup_mp, define_model_in_strategy, get_callbacks, init_custom_checkpoint_callbacks
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
    ['list[str]', 'frameworks', ['tensorflow'], 'List of framework to use to define the model', lambda x: not any(y not in framework_list for y in x)],
    ['namespace', 'factor', [[str, 'scale', 'log', 'linear or log'], [float, 'min', 1, ''], [float, 'max', 1, ''], [float, 'step', 0, ''], ]],
    [str, "model_name", '', 'Specify the name of the model', lambda x: x in model_name_to_class],

] + global_conf.arguments + training.arguments


def main():
  """ function called when starting the code via command-line
  """
  args = argparse.parse_cmd(arguments)
  args['server'] = alchemy_api.start_training(args['server'])
  train(args)


def get_values_from_args(args):
  if args['scale'] == 'linear':
    values = list(range(args['min'], args['max'], args['step']))
  elif args['scale'] == 'log':
    values = []
    previous = args['min']
    while previous <= args['max']:
      values.append(previous)
      previous *= args['step']
  else:
    raise ValueError(f"unknown scale '{args['scale']}'")
  return values


def get_model(args):
  def build_model(hp):
    factor = hp.Choice('factor', get_values_from_args(args['factor']), ordered=True)
    framework = hp.Choice('framework', args['frameworks'])
    model = model_name_to_class[args['model_name']](framework,
                                                    factor,
                                                    args['input_size'],
                                                    args['num_classes'],
                                                    hp=hp).model
    model.compile(
        optimizer=get_optimizer(args['optimizer']),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
  return build_model


def get_experiment_name(args):
  experiment_dir = f"keras_tuner_{args['model_name']}"
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
  build_model_fn = get_model(args)
  callbacks = get_callbacks(args, log_dir)

  # tuner = Hyperband(build_model_fn,
  #                   objective='val_accuracy',
  #                   max_epochs=args['num_epochs'],
  #                   hyperband_iterations=10e100,
  #                   directory=checkpoint_dir)

  tuner = BayesianOptimization(build_model_fn,
                    objective='val_accuracy',
                    max_trials=100000,
                    num_initial_points=10,
                    directory=checkpoint_dir)

  tuner.search_space_summary()
  tuner.search(x=train_dataset,
               validation_data=val_dataset,
               callbacks=callbacks,
               epochs=args['num_epochs'])
  tuner.results_summary()


if __name__ == '__main__':
  main()
