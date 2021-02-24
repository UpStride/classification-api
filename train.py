import tensorflow as tf
import os
import yaml
import upstride_argparse as argparse
from src.data import dataloader
from src.models import model_name_to_class
from src.utils import check_folder, get_imagenet_data, model_dir
from submodules.global_dl import global_conf
from submodules.global_dl.training.training import create_env_directories, get_callbacks, init_custom_checkpoint_callbacks
from submodules.global_dl.training import training
from submodules.global_dl.training import alchemy_api
from submodules.global_dl.training import export
from submodules.global_dl.training.optimizers import get_lr_scheduler, get_optimizer, arguments
from submodules.global_dl.training import optimizers
from submodules.global_dl.training import metrics


arguments = [
    ['namespace', 'dataloader', dataloader.arguments],
    ['namespace', 'server', alchemy_api.arguments],
    ['namespace', 'optimizer', optimizers.arguments],
    ['namespace', 'export', export.arguments],
    ['namespace', 'config', global_conf.arguments],
    [str, 'load_searched_arch', '', 'model definition file containing the searched architecture', ],
    [str, 'experiment_name', '', 'model definition file containing the searched architecture', ],
    ['namespace', 'model', [
        [int, 'upstride_type', -1, 'if set to a number in [0, 3] then use this upstride type', lambda x: x < 4 and x > -2],
        [int, "factor", 1, 'division factor to scale the number of channel. factor=2 means the model will have half the number of channels compare to default implementation'],
        ['list[int]', "input_size", [224, 224, 3], 'processed shape of each image'],
        ['list[str]', "changing_ids", ['beginning', 'end_after_dense'], 'id corresponding to the framework changes'],
        # [int, 'n_layers_before_tf', 0, 'when using mix framework, number of layer defined using upstride', lambda x: x >= 0],
        [str, "name", '', 'Specify the name of the model', lambda x: x in model_name_to_class],
        [float, "drop_path_prob", 0.3, 'drop path probability'],
        [int, "num_classes", 0, 'Number of classes', lambda x: x > 0],  # TODO this number should be computed from dataset
        [bool, "channels_first", False, 'Start training with channels first'],
        [bool, "calculate_flops", False, 'when true displays the total FLOPs for the given model'], # TODO can be removed when issue with type1 is fixed
        ['namespace', 'conversion_params', [
            # [bool, 'output_layer_before_up2tf', False, 'Whether to use final output layer before UpStride2TF conversion or not'],
            [str, 'tf2up_strategy', '', 'TF2UpStride conversion strategy'],
            [str, 'up2tf_strategy', 'default', 'UpStride2TF conversion strategy']
        ]],
    ]],

] + training.arguments


def get_compiled_model(config):

  # Parameters usefull for all the models
  kwargs = {
    'input_size': config['model']['input_size'],
    'changing_ids': config['model']['changing_ids'],
    'num_classes': config['model']['num_classes'],
    'factor': config['model']['factor'],
    'upstride_type': config['model']['upstride_type'],
    'tf2upstride_strategy': config['model']['conversion_params']['tf2up_strategy'],
    'upstride2tf_strategy': config['model']['conversion_params']['up2tf_strategy'],
    'weight_decay': config['optimizer']['weight_decay'],
  }

  # for architecture search
  if config['load_searched_arch']:
    kwargs['load_searched_arch'] = config['load_searched_arch']

  model = model_name_to_class[config['model']['name']](**kwargs).build()
  model.summary()
  # calculates FLOPs for the Model.
  if config['model']['calculate_flops']:
    calc_flops = metrics.count_flops_efficient(model, config['model']['upstride_type'])
    print(f"Total FLOPs for {config['model']['name']}: {calc_flops}")
  optimizer = get_optimizer(config['optimizer'])
  model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                metrics=['accuracy', 'top_k_categorical_accuracy'])
  # output the optimizer to save it in the checkpoint
  return model, optimizer


def get_experiment_name(params):
  """if experiment_name is defined in the parameters, then return it. Else create a custom experiment_name based
  on the other parameters

  Returns: [str] the experiment name
  """
  if params['experiment_name']:
    return params['experiment_name']

  args_model = params['model']

  upstride_type = args_model['upstride_type']
  framework = 'tensorflow' if upstride_type == -1 else f'upstride_{upstride_type}'

  experiment_dir = f"{args_model['name']}_{framework}_factor{args_model['factor']}"
  experiment_dir += "_NCHW" if args_model['channels_first'] else "_NHWC"
  if params['config']['mixed_precision']:
    experiment_dir += "_mp"
  return experiment_dir


def train(config):
  """
  This function setup:
    1- Tensorflow (XLA, GPU configuration, mixed precision, execution strategies)
    2- The datasets
    3- The model
    4- The execution environment
    5- The monitoring (Upstride plateform and tensorboard)

  Then 
    6- start the training
    7- Export the model
  """

  # 1
  global_conf.config_tf2(config['config'])
  global_conf.setup_mp(config['config'])
  ds_strategy = global_conf.setup_strategy(config['config']['strategy'])
  if config['model']['channels_first']:  # if True set keras backend to channels_first
    tf.keras.backend.set_image_data_format('channels_first')

  # 2
  train_dataset = dataloader.get_dataset(config['dataloader'], transformation_list=config['dataloader']['train_list'],
                                         num_classes=config['model']["num_classes"], split=config['dataloader']['train_split_id'])
  val_dataset = dataloader.get_dataset(config['dataloader'], transformation_list=config['dataloader']['val_list'],
                                       num_classes=config['model']["num_classes"], split=config['dataloader']['val_split_id'])

  # 3
  with ds_strategy.scope():
    model, optimizer = get_compiled_model(config)

  # 4
  checkpoint_dir, log_dir, export_dir = create_env_directories(get_experiment_name(config), config['checkpoint_dir'], config['log_dir'], config['export']['dir'])
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  with open(os.path.join(log_dir, "conf.yml"), 'w') as file:
    yaml.dump(config, file)

  # 5
  config['server'] = alchemy_api.start_training(config['server'])
  alchemy_api.send_model_info(model, config['server'])
  callbacks = get_callbacks(config, log_dir)

  with ds_strategy.scope(): # checkpoints needs to be in the same scope.
    model_checkpoint_cb, latest_epoch = init_custom_checkpoint_callbacks({'model': model}, checkpoint_dir, config['max_checkpoints'], config['checkpoint_freq'])

  callbacks.append(model_checkpoint_cb)
  if config['server']['id'] != '':
    callbacks.append(alchemy_api.send_metric_callbacks(config['server']))

  if config['model']['name'] == 'Pdart':
    from src.models.pdart import callback_epoch
    callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch, logs: callback_epoch(epoch, config['num_epochs'], config['drop_path_prob'])))

  # 6 training
  model.fit(x=train_dataset,
            validation_data=val_dataset,
            epochs=config['num_epochs'],
            callbacks=callbacks,
            max_queue_size=16,
            initial_epoch=latest_epoch
            )

  # 7 training
  print("export model")
  export.export(model, export_dir, config)
  print("Training Completed!!")


if __name__ == '__main__':
  config = argparse.parse_cmd(arguments)
  train(config)
