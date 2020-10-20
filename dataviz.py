import cv2
import os
import tensorflow as tf
import upstride_argparse as argparse
from src.data import dataloader

arguments = [
    ['namespace', 'dataloader', dataloader.arguments],
]


def main():
  args = argparse.parse_cmd(arguments)
  datasets = {
      'train': dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['train_list'], num_classes=10, split='train'),
      'val': dataloader.get_dataset(args['dataloader'], transformation_list=args['dataloader']['val_list'], num_classes=10, split='test')
  }
  for dataset_type in ['train', 'val']:
    for i, (images, y) in enumerate(datasets[dataset_type]):
      image = images[0]
      # opencv manage images as BGR object, TF as RGB
      image = image.numpy()[:, :, ::-1]
      cv2.imwrite(os.path.join('/tmp', f'{dataset_type}_{i}.png'), image)
      if i == 20:
        break


if __name__ == '__main__':
  main()
