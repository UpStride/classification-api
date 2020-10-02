import os
import json
import sys
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plot = True
try:
  matplotlib.use("GTK3Agg")
except ImportError:
  print("can't load matplotlib")
  plot = False



def main(csv_path):
  if not os.path.exists(csv_path):
    all_data = []
    all_keys = ['score']
    for d in os.listdir('.'):
      if not os.path.isdir(d):
        continue
      with open(os.path.join(d, 'trial.json'), 'r') as f:
        json_data = json.load(f)
      data = json_data['hyperparameters']['values']
      data['score'] = json_data['score']
      data['name'] = d
      all_data.append(data)
      for key in data:
        if key not in all_keys:
          all_keys.append(key)
    # write csv
    with open(csv_path, 'w') as f:
      f.write(",".join(all_keys))
      f.write("\n")
      for data in all_data:
        to_write = []
        for key in all_keys:
          if key in data:
            if data[key] is None:
              data[key] = 0
            to_write.append(str(data[key]))
          else:
            to_write.append('')
        f.write(','.join(to_write))
        f.write('\n')
    print(f"file {csv_path} written")
  # also start a visu with seaborn

  if not plot:
    return
  sns.set(style="darkgrid")
  data = pd.read_csv(csv_path)

  # data['depth'] = data['conv3_depth'] + data['conv4_depth']
  data['depth'] *= 2
  depths = data['depth'].unique()
  depths.sort()
  factors = data['factor'].unique()
  factors.sort()
  if 'tuner/trial_id' in data:
    data = data.drop(columns=['tuner/trial_id'])

  g = sns.FacetGrid(data, col="tuner/epochs", legend_out=True)
  g.map_dataframe(draw_heatmap, 'factors', 'depths', factors=factors, depths=depths)
  g.add_legend()
  plt.show()

def draw_heatmap(*args, **kwargs):
  '''from https://stackoverflow.com/questions/41471238/how-to-make-heatmap-square-in-seaborn-facetgrid
  '''
  data = kwargs.pop('data')
  data['score'] = data['score'] * 100 
  data = data.pivot(index='depth', columns='factor', values='score')
  # add missing factor
  for f in kwargs['factors']:
    if f not in data:
      data[f] = np.nan
  # add missing indexes
  index_to_add = []    
  for d in kwargs['depths']:
    if d not in data.index:
      index_to_add.append(d)
  new_indexes = data.index.values.tolist() + index_to_add
  new_indexes.sort()      
  data = data.reindex(new_indexes)
  data = data.reindex(sorted(data.columns), axis=1)

  sns.heatmap(data, annot=True, vmin=0, vmax=100, cmap='CMRmap')

  # sns.heatmap(matrix, annot=True, linewidth=0.5, xticklabels=factors, yticklabels=depths, vmin=0, vmax=100, cmap='CMRmap', mask=mask)


if __name__ == "__main__":
  csv_path = sys.argv[1]
  main(csv_path)
