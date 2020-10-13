import tqdm
import yaml
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Mean
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


arguments = [
    ['namespace', 'dataloader', dataloader.arguments],
    ['namespace', 'server', alchemy_api.arguments],
    ['namespace', 'optimizer', optimizers.arguments],
    ['namespace', 'export', export.arguments],
    [int, "factor", 1, 'division factor to scale the number of channel. factor=2 means the model will have half the number of channels compare to default implementation'],
    [int, 'n_layers_before_tf', 0, 'when using mix framework, number of layer defined using upstride', lambda x: x >= 0],
    [str, 'load_searched_arch', '', 'model definition file containing the searched architecture'],
    [str, "model_name", '', 'Specify the name of the model', lambda x: x in model_name_to_class],
    [str, 'framework', 'tensorflow', 'Framework to use to define the model', lambda x: x in framework_list],
] + global_conf.arguments + training.arguments + training_arguments_das


def exponential_decay(initial_value, decay_steps, decay_rate):
  """
          Applies exponential decay to initial value
       Args:
          initial_value: The initial learning value
          decay_steps: Number of steps to decay over
          decay_rate: decay rate
      """
  return lambda step: initial_value * decay_rate ** (step / decay_steps)


def split_trainable_weights(model):
  """
      split the model parameters  in weights and architectural params
  """
  weights = []
  arch_params = []
  for trainable_weight in model.trainable_variables:
    if 'alpha' in trainable_weight.name:
      arch_params.append(trainable_weight)
    else:
      weights.append(trainable_weight)

  return model.trainable_variables, arch_params


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
  train_accuracy_metric = metrics['train_accuracy']
  train_cross_entropy_loss_metric = metrics['train_cross_entropy_loss']
  train_total_loss_metric = metrics['train_total_loss']

  @tf.function
  def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
      y_hat = model(x_batch, training=True)
      cross_entropy_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, y_hat))
      # TODO not sure of this computation. I think we should use model.losses to get the internal losses (l2) Doing this will allow the user to define both L1 and L2 reg in the model
      weight_reg_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights if 'bias' not in w.name])
      total_loss = cross_entropy_loss + args['weight_decay'] * weight_reg_loss
    train_accuracy_metric.update_state(y_batch, y_hat)
    train_cross_entropy_loss_metric.update_state(cross_entropy_loss)
    train_total_loss_metric.update_state(total_loss)
    # Update the weights
    grads = tape.gradient(total_loss, weights)
    weight_opt.apply_gradients(zip(grads, weights))
  return train_step


def get_train_step_arch_function(model, arch_params, arch_opt, metrics):
  latency_reg_loss_metric = metrics['arch_latency_reg_loss']
  train_accuracy_metric = metrics['train_accuracy']
  train_cross_entropy_loss_metric = metrics['train_cross_entropy_loss']
  total_loss_metric = metrics['train_total_loss']

  @tf.function
  def train_step_arch(x_batch, y_batch):
    with tf.GradientTape() as tape:
      y_hat = model(x_batch, training=False)
      cross_entropy_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, y_hat))
      # TODO do we want L2 reg for arch parameters ?
      weight_reg_loss = tf.add_n([tf.nn.l2_loss(w) for w in arch_params if 'bias' not in w.name])
      latency_reg_loss = losses.parameters_loss(model) / 1.0e6
      total_loss = cross_entropy_loss + args['arch_param_decay'] * weight_reg_loss + latency_reg_loss
    latency_reg_loss_metric.update_state(latency_reg_loss)
    train_accuracy_metric.update_state(y_batch, y_hat)
    train_cross_entropy_loss_metric.update_state(cross_entropy_loss)
    total_loss_metric.update_state(total_loss)
    # Update the architecture paramaters
    grads = tape.gradient(total_loss, arch_params)
    arch_opt.apply_gradients(zip(grads, arch_params))
  return train_step_arch


def get_eval_step_function(model, metrics):
  val_accuracy_metric = metrics['val_accuracy']
  val_cross_entropy_loss_metric = metrics['val_cross_entropy_loss']

  @tf.function
  def evaluation_step(x_batch, y_batch):
    y_hat = model(x_batch, training=False)
    loss = loss_fn(y_batch, y_hat)
    val_accuracy_metric.update_state(y_batch, y_hat)
    val_cross_entropy_loss_metric.update_state(loss)
  return evaluation_step


def train(args):
  # config_tf2(args['configuration']['xla'])
  # Create log, checkpoint and export directories
  checkpoint_dir, log_dir, export_dir = create_env_directories(args, get_experiment_name(args))

  train_weight_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['train_list'],
                                                num_classes=args["num_classes"], split='train_weights')
  train_arch_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['train_list'],
                                              num_classes=args["num_classes"], split='train_arch')
  val_dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['val_list'],
                                       num_classes=args["num_classes"], split='test')

  setup_mp(args)

  # define model, optimizer and checkpoint callback
  model = model_name_to_class[args['model_name']](args['framework'],
                                                  input_shape=args['input_size'],
                                                  label_dim=args['num_classes']).model
  model.summary()
  alchemy_api.send_model_info(model, args['server'])
  weight_opt = get_optimizer(args['optimizer'])
  arch_opt = get_optimizer(args['arch_optimizer_param'])
  model_checkpoint_cb, latest_epoch = init_custom_checkpoint_callbacks({'model': model}, checkpoint_dir)

  weights, arch_params = split_trainable_weights(model)
  temperature_decay_fn = exponential_decay(args['temperature']['init_value'],
                                           args['temperature']['decay_steps'],
                                           args['temperature']['decay_rate'])

  lr_decay_fn = CosineDecay(args['optimizer']['lr'],
                            alpha=args["optimizer"]["lr_decay_strategy"]["lr_params"]["alpha"],
                            total_epochs=args['num_epochs'])

  lr_decay_fn_arch = CosineDecay(args['arch_optimizer_param']['lr'],
                                 alpha=0.000001,
                                 total_epochs=args['num_epochs'])

  metrics = {
      'arch_latency_reg_loss': tf.keras.metrics.Mean()
      'train_total_loss': tf.keras.metrics.Mean(),
      'train_accuracy': tf.keras.metrics.CategoricalAccuracy(),
      'train_cross_entropy_loss': tf.keras.metrics.Mean(),
      'val_accuracy': tf.keras.metrics.CategoricalAccuracy(),
      'val_cross_entropy_loss': tf.keras.metrics.Mean(),
  }

  train_step = get_train_step_function(model, weights, weight_opt, metrics)
  train_step_arch = get_train_step_arch_function(model, arch_params, arch_opt, metrics)
  evaluation_step = get_eval_step_function(model, metrics)

  for epoch in range(latest_epoch, args['num_epochs']):
    print(f'Epoch: {epoch}/{args["num_epochs"]}')

    weight_opt.learning_rate = lr_decay_fn(epoch)
    arch_opt.learning_rate = lr_decay_fn_arch(epoch)

    # Updating the weight parameters using a subset of the training data
    for step, (x_batch, y_batch) in tqdm.tqdm(enumerate(train_weight_dataset, start=1)):
      train_step(x_batch, y_batch)

    # Evaluate the model on validation subset
    for x_batch, y_batch in val_dataset:
      evaluation_step(x_batch, y_batch)

    train_accuracy = train_accuracy_metric.result()
    train_cross_entropy_loss = train_cross_entropy_loss_metric.result()
    train_total_loss = total_loss_metric.result()
    val_accuracy = val_accuracy_metric.result()
    val_cross_entropy_loss = val_cross_entropy_loss_metric.result()

    template = f'Weights updated, Epoch {epoch}, Train total Loss: {float(train_total_loss)}, ' \
        f'Train Cross Entropy Loss: {float(train_cross_entropy_loss)}, ' \
        f'Train Accuracy: {float(train_accuracy)} Val Loss: {float(val_cross_entropy_loss)}, ' \
        f'Val Accuracy: {float(val_accuracy)}, lr: {float(weight_opt.learning_rate)}'
    print(template)

    new_temperature = temperature_decay_fn(epoch)

    with train_summary_writer.as_default():
      tf.summary.scalar('total loss', train_total_loss, step=epoch)
      tf.summary.scalar('cross entropy loss', train_cross_entropy_loss, step=epoch)
      tf.summary.scalar('accuracy', train_accuracy, step=epoch)
      tf.summary.scalar('temperature', new_temperature, step=epoch)

    with val_summary_writer.as_default():
      tf.summary.scalar('cross entropy loss', val_cross_entropy_loss, step=epoch)
      tf.summary.scalar('accuracy', val_accuracy, step=epoch)

    # Resetting metrices for reuse
    train_accuracy_metric.reset_states()
    train_cross_entropy_loss_metric.reset_states()
    total_loss_metric.reset_states()
    val_accuracy_metric.reset_states()
    val_cross_entropy_loss_metric.reset_states()

    if epoch >= args['num_warmup']:
      # Updating the architectural parameters on another subset
      for step, (x_batch, y_batch) in tqdm.tqdm(enumerate(train_arch_dataset, start=1)):
        train_step_arch(x_batch, y_batch)

      # Evaluate the model on validation subset
      for x_batch, y_batch in val_dataset:
        evaluation_step(x_batch, y_batch)

      train_accuracy = train_accuracy_metric.result()
      train_total_loss = total_loss_metric.result()
      train_cross_entropy_loss = train_cross_entropy_loss_metric.result()
      val_accuracy = val_accuracy_metric.result()
      val_loss = val_cross_entropy_loss_metric.result()
      latency_reg_loss = latency_reg_loss_metric.result()

      template = f'Weights updated, Epoch {epoch}, Train total Loss: {float(train_total_loss)}, ' \
          f'Train Cross Entropy Loss: {float(train_cross_entropy_loss)}, ' \
          f'Train Accuracy: {float(train_accuracy)} Val Loss: {float(val_loss)}, Val Accuracy: {float(val_accuracy)}, ' \
          f'reg_loss: {float(latency_reg_loss)}'
      print(template)
      with train_summary_writer.as_default():
        tf.summary.scalar('total_loss_after_arch_params_update', train_total_loss, step=epoch)
        tf.summary.scalar('cross_entropy_loss_after_arch_params_update', train_cross_entropy_loss, step=epoch)
        tf.summary.scalar('accuracy_after_arch_params_update', train_accuracy, step=epoch)
        tf.summary.scalar('latency_reg_loss', latency_reg_loss, step=epoch)

      with val_summary_writer.as_default():
        tf.summary.scalar('total_loss_after_arch_params_update', val_loss, step=epoch)
        tf.summary.scalar('accuracy_after_arch_params_update', val_accuracy, step=epoch)

      # Resetting metrices for reuse
      train_accuracy_metric.reset_states()
      train_cross_entropy_loss_metric.reset_states()
      total_loss_metric.reset_states()
      val_accuracy_metric.reset_states()
      val_cross_entropy_loss_metric.reset_states()

    define_temperature(new_temperature)

  print("Training Completed!!")

  print("Architecture params: ")
  print(arch_params)
  post_training_analysis(model, args['exported_architecture'])


if __name__ == '__main__':
  main()
