import math
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import upstride_argparse as argparse

arguments = [
    [str, "alpha_path", '', 'path of the alpha file to parse'],
    [int, "epoch", 0, 'if different than 0 then visualize a single epoch'],
    [int, "min", 0, 'if provided, then define the minimum epoch to visualize'],
    [int, "max", 0, 'if provided, then define the maximum epoch to visualize'],
    [int, "step", 100, 'number of steps between 2 epochs to visualize'],
    ['list[str]', "params", [], 'if specified, list of parameters to  visualize']
]


def prepare_data(args):
  with open(args['alpha_path'], 'r') as f:
    alphas = json.load(f)
  # find min and max epochs
  epochs = list(map(int, alphas.keys()))
  epochs.sort()

  min_epoch = max(epochs[0], args['min']) if args['min'] else epochs[0]
  max_epoch = min(epochs[-1], args['max']) if args['max'] else epochs[-1]

  if args['epoch']:
    min_epoch = args['epoch']
    max_epoch = args['epoch']

  # find parameters to visualize, and remove final '_savable'
  params = alphas[str(min_epoch)].keys()
  params = list(map(lambda x: x[:-8], alphas[str(min_epoch)].keys()))
  if args['params']:
    params = args['params']
  print("display:", params)
  print('epochs:', min_epoch, max_epoch)
  return alphas, min_epoch, max_epoch, params


def main():
  matplotlib.use("GTK3Agg")
  args = argparse.parse_cmd(arguments)
  alphas, min_epoch, max_epoch, params = prepare_data(args)
  number_bars = (max_epoch - min_epoch) // args["step"] + 1
  colors = cm.OrRd_r(np.linspace(.2, .6, number_bars))

  # grid has a fixed number of columns of 5
  n_params = len(params)
  if n_params == 1:
    fig, axs = plt.subplots(1, 1, figsize=(9, 3))
    axs = [[axs]]
  elif n_params <= 5:
    fig, axs = plt.subplots(math.ceil(n_params), 1, figsize=(9, 3))
    axs = [axs]
  else:
    fig, axs = plt.subplots(math.ceil(n_params/5), 5, figsize=(9, 3))
  fig.suptitle(f'Alpha  parameter between {min_epoch} and {max_epoch} epochs (step: {args["step"]})')

  total_width = 0.7
  width = total_width / number_bars
  for i in range(number_bars):
    epoch = min_epoch + i * args["step"]
    for k, param in enumerate(params):
      p = alphas[str(epoch)][param + '_savable']

      # TODO should be removed as soon as data is better formated
      data = {}
      for j in range(len(p)):
        data[str(j)] = p[j]

      names = list(data.keys())
      values = list(data.values())

      x = np.arange(len(p))
      axs[k//5][k % 5].bar(x - total_width/2 + width * i, values, width, label=str(i)) #, color=colors)
      axs[k//5][k%5].set_title(param)
      
  # fig.tight_layout()
  plt.show()


if __name__ == "__main__":
  main()
