import os
import tensorflow as tf
import upstride_argparse as argparse
from src.data import dataloader, augmentations
from src.models.generic_model import framework_list
from src.models import model_name_to_class
from submodules.global_dl import global_conf


args_spec = [
    ['list[str]', "yaml_config", [], "config file overriden by these argparser parameters"],

    # framework specification
    [str, 'framework', 'tensorflow', 'Framework to use to define the model', lambda x: x in framework_list],

    # model specification
    [str, "model_name", None, 'Specify the name of the model', lambda x: x in model_name_to_class],
    [int, "factor", 1, 'division factor to scale the number of channel. factor=2 means the model will have half the number of channels compare to default implementation'],
    [int, 'n_layers_before_tf', 0, 'when using mix framework, number of layer defined using upstride', lambda x: x >= 0],
    [int, "num_classes", 0, 'Number of classes', lambda x: x > 0],
    ['list[int]', "input_size", [224, 224, 3], 'processed shape of each image'],

    # checkpoints directory
    [str, "checkpoint_dir", None, 'Checkpoints directory to load the trained model from'],

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
] + global_conf.arguments


def main():
  """ CLI entry point
  """
  # parse arguments
  args = argparse.parse_cmd(args_spec)

  # perform global configuration (XLA and memory growth)
  global_conf.config_tf2(args)

  # instantiate model
  model = model_name_to_class[args['model_name']](args['framework'],
                                                  args['factor'],
                                                  args['input_size'],
                                                  args['num_classes'],
                                                  args['n_layers_before_tf'], False).model
  model.compile(loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
  model.summary()

  # load a checkpoint
  from train import get_experiment_name
  ckpt_dir = os.path.join(args['checkpoint_dir'], get_experiment_name(args))
  checkpoint = tf.train.Checkpoint(model=model)
  manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir, max_to_keep=5)
  restored_ckpt = manager.restore_or_initialize()
  if restored_ckpt is None:
    raise RuntimeError(f"Cannot restore from a checkpoint in {ckpt_dir}")
  print(f'Restoring {manager.latest_checkpoint}')

  # if dataloader.name is set, evaluating on a specific dataset
  if args['dataloader']['name'] is not None:
    print(f"Evaluating on {args['dataloader']['name']}")
    args['dataloader']['train_split_id'] = None
    dataset = dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['list'],
                                     num_classes=args["num_classes"], split=args['dataloader']['split_id'])
    model.evaluate(dataset)

if __name__ == '__main__':
  main()
