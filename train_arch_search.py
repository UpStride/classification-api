import tqdm
import yaml
from src.models.fbnetv2 import ChannelMasking, define_temperature
from submodules.global_dl.global_conf import config_tf2
import math
import os
import tensorflow as tf
import upstride_argparse as argparse
from src.argument_parser import training_arguments_das
from src.data import dataloader
from src import losses
from src.models import model_name_to_class
from src.models.generic_model import framework_list
from src.utils import check_folder, get_imagenet_data, model_dir
from submodules.global_dl import global_conf
from submodules.global_dl.training.training import create_env_directories, setup_mp, define_model_in_strategy, get_callbacks, init_custom_checkpoint_callbacks
from submodules.global_dl.training import training
from submodules.global_dl.training import alchemy_api
from submodules.global_dl.training import export
from submodules.global_dl.training.optimizers import get_lr_scheduler, get_optimizer, StepDecaySchedule, CosineDecay
from submodules.global_dl.training import optimizers
from src.models import fbnetv2

arguments = [
    ['namespace', 'dataloader', dataloader.arguments],
    ['namespace', 'server', alchemy_api.arguments],
    ['namespace', 'optimizer', optimizers.arguments],
    ['namespace', 'export', export.arguments],
    ['namespace', 'arch_search', training_arguments_das],
    [int, "factor", 1, 'division factor to scale the number of channel. factor=2 means the model will have half the number of channels compare to default implementation'],
    [int, 'n_layers_before_tf', 0, 'when using mix framework, number of layer defined using upstride', lambda x: x >= 0],
    [str, 'load_searched_arch', '', 'model definition file containing the searched architecture'],
    [bool, 'log_arch', False, 'if true then save the values of the alpha parameters after every epochs in a csv file in log directory'],
    [str, "model_name", '', 'Specify the name of the model', lambda x: x in model_name_to_class],
    [str, 'framework', 'tensorflow', 'Framework to use to define the model', lambda x: x in framework_list],
] + global_conf.arguments + training.arguments


def main():
  """ function called when starting the code via command-line
  """
  args = argparse.parse_cmd(arguments)
  args['server'] = alchemy_api.start_training(args['server'])
  train(args)


def get_experiment_name(args):
  experiment_dir = f"{args['model_name']}_{args['framework']}"
  if 'mix' in args['framework']:
    experiment_dir += "_mix_{}".format(args['n_layers_before_tf'])
  if args['configuration']['with_mixed_precision']:
    experiment_dir += "_mp"
  return experiment_dir


def get_train_step_function(model, weights, weight_opt, metrics):
  train_accuracy_metric = metrics['accuracy']
  train_cross_entropy_loss_metric = metrics['cross_entropy_loss']
  train_total_loss_metric = metrics['total_loss']

  @tf.function
  def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
      y_hat = model(x_batch, training=True)
      cross_entropy_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, y_hat))
      weight_reg_loss = tf.reduce_sum(model.losses)
      total_loss = cross_entropy_loss + weight_reg_loss
    train_accuracy_metric.update_state(y_batch, y_hat)
    train_cross_entropy_loss_metric.update_state(cross_entropy_loss)
    train_total_loss_metric.update_state(total_loss)
    # Update the weights
    grads = tape.gradient(total_loss, weights)
    weight_opt.apply_gradients(zip(grads, weights))
  return train_step


def get_train_step_arch_function(model, arch_params, arch_opt, train_metrics, arch_metrics):
  latency_reg_loss_metric = arch_metrics['latency_reg_loss']
  train_accuracy_metric = train_metrics['accuracy']
  train_cross_entropy_loss_metric = train_metrics['cross_entropy_loss']
  total_loss_metric = train_metrics['total_loss']

  @tf.function
  def train_step_arch(x_batch, y_batch):
    with tf.GradientTape() as tape:
      y_hat = model(x_batch, training=False)
      cross_entropy_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, y_hat))
      weight_reg_loss = tf.reduce_sum(model.losses)
      latency_reg_loss = losses.parameters_loss(model) / 1.0e6
      total_loss = cross_entropy_loss + weight_reg_loss  # + latency_reg_loss
    latency_reg_loss_metric.update_state(latency_reg_loss)
    train_accuracy_metric.update_state(y_batch, y_hat)
    train_cross_entropy_loss_metric.update_state(cross_entropy_loss)
    total_loss_metric.update_state(total_loss)
    # Update the architecture paramaters
    grads = tape.gradient(total_loss, arch_params)
    arch_opt.apply_gradients(zip(grads, arch_params))
  return train_step_arch


def get_eval_step_function(model, metrics):
  val_accuracy_metric = metrics['accuracy']
  val_cross_entropy_loss_metric = metrics['cross_entropy_loss']

  @tf.function
  def evaluation_step(x_batch, y_batch):
    y_hat = model(x_batch, training=False)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, y_hat))
    val_accuracy_metric.update_state(y_batch, y_hat)
    val_cross_entropy_loss_metric.update_state(loss)
  return evaluation_step


def metrics_processing(metrics, summary_writers, keys, template, epoch, postfix=''):
  for key in keys:
    with summary_writers[key].as_default():
      for sub_key in metrics[key]:
        value = float(metrics[key][sub_key].result())  # save metric value
        metrics[key][sub_key].reset_states()  # reset the metric
        template += f", {key}_{sub_key}: {value}"
        tf.summary.scalar(sub_key+postfix, value, step=epoch)
  return template


def train(args):
  # config_tf2(args['configuration']['xla'])
  # Create log, checkpoint and export directories
  checkpoint_dir, log_dir, export_dir = create_env_directories(args, get_experiment_name(args))
  train_log_dir = os.path.join(log_dir, 'train')
  val_log_dir = os.path.join(log_dir, 'validation')
  arch_log_dir = os.path.join(log_dir, 'arch')
  summary_writers = {
      'train': tf.summary.create_file_writer(train_log_dir),
      'val': tf.summary.create_file_writer(val_log_dir),
      'arch': tf.summary.create_file_writer(arch_log_dir)
  }

  # Prepare the 3 datasets
  train_weight_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['train_list'],
                                                num_classes=args["num_classes"], split='train_weights')
  train_arch_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['train_list'],
                                              num_classes=args["num_classes"], split='train_arch')
  val_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['val_list'],
                                       num_classes=args["num_classes"], split='test')

  # define model, optimizer and checkpoint callback
  setup_mp(args)
  model = model_name_to_class[args['model_name']](args['framework'],
                                                  input_shape=args['input_size'],
                                                  label_dim=args['num_classes']).model
  model.summary()

  alchemy_api.send_model_info(model, args['server'])
  weights, arch_params = fbnetv2.split_trainable_weights(model)
  weight_opt = get_optimizer(args['optimizer'])
  arch_opt = get_optimizer(args['arch_search']['optimizer'])
  model_checkpoint_cb, latest_epoch = init_custom_checkpoint_callbacks({'model': model}, checkpoint_dir)
  callbacks = [
      model_checkpoint_cb
  ]

  temperature_decay_fn = fbnetv2.exponential_decay(args['arch_search']['temperature']['init_value'],
                                                   args['arch_search']['temperature']['decay_steps'],
                                                   args['arch_search']['temperature']['decay_rate'])

  lr_decay_fn = CosineDecay(args['optimizer']['lr'],
                            alpha=args["optimizer"]["lr_decay_strategy"]["lr_params"]["alpha"],
                            total_epochs=args['num_epochs'])

  lr_decay_fn_arch = CosineDecay(args['arch_search']['optimizer']['lr'],
                                 alpha=0.000001,
                                 total_epochs=args['num_epochs'])

  metrics = {
      'arch': {
          'latency_reg_loss': tf.keras.metrics.Mean()
      },
      'train': {
          'total_loss': tf.keras.metrics.Mean(),
          'accuracy': tf.keras.metrics.CategoricalAccuracy(),
          'cross_entropy_loss': tf.keras.metrics.Mean(),
      },
      'val': {
          'accuracy': tf.keras.metrics.CategoricalAccuracy(),
          'cross_entropy_loss': tf.keras.metrics.Mean(),
      }
  }

  train_step = get_train_step_function(model, weights, weight_opt, metrics['train'])
  train_step_arch = get_train_step_arch_function(model, arch_params, arch_opt, metrics['train'], metrics['arch'])
  evaluation_step = get_eval_step_function(model, metrics['val'])

  for epoch in range(latest_epoch, args['num_epochs']):
    print(f'Epoch: {epoch}/{args["num_epochs"]}')
    # Update both LR
    weight_opt.learning_rate = lr_decay_fn(epoch)
    arch_opt.learning_rate = lr_decay_fn_arch(epoch)
    # Updating the weight parameters using a subset of the training data
    for step, (x_batch, y_batch) in tqdm.tqdm(enumerate(train_weight_dataset, start=1)):
      train_step(x_batch, y_batch)
    # Evaluate the model on validation subset
    for x_batch, y_batch in val_dataset:
      evaluation_step(x_batch, y_batch)
    # Handle metrics
    template = f"Weights updated, Epoch {epoch}"
    template = metrics_processing(metrics, summary_writers, ['train', 'val'], template, epoch)
    template += f", lr: {float(weight_opt.learning_rate)}"
    print(template)

    new_temperature = temperature_decay_fn(epoch)
    with summary_writers['train'].as_default():
      tf.summary.scalar('temperature', new_temperature, step=epoch)
    define_temperature(new_temperature)

    if epoch >= args['arch_search']['num_warmup']:
      # Updating the architectural parameters on another subset
      for step, (x_batch, y_batch) in tqdm.tqdm(enumerate(train_arch_dataset, start=1)):
        train_step_arch(x_batch, y_batch)
      # Evaluate the model on validation subset
      for x_batch, y_batch in val_dataset:
        evaluation_step(x_batch, y_batch)
      # Handle metrics
      template = f'Architecture updated, Epoch {epoch}'
      template = metrics_processing(metrics, summary_writers, ['train', 'val', 'arch'], template, epoch, postfix='_arch')
      template += f", lr: {float(arch_opt.learning_rate)}"
      print(template)
    # move saved outside of condition so we save starting from the begining
    fbnetv2.save_arch_params(model, epoch, log_dir)

    # manually call the callbacks
    for callback in callbacks:
      callback.on_epoch_end(epoch, logs=None)

  print("Training Completed!!")

  print("Architecture params: ")
  print(arch_params)
  fbnetv2.post_training_analysis(model, args['arch_search']['exported_architecture'])


if __name__ == '__main__':
  main()
