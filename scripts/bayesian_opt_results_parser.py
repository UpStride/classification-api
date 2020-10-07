import os
import json
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import upstride_argparse as argparse

arguments = [
    [str, "server", '', 'address of the server to connect using ssh'],
    [str, 'remote_dir', '', "directory of the keras tuner experiment on the remote server"],
    [str, 'csv_path', '/tmp/results.csv', 'path to write csv file'],
    [bool, 'no_plot', False, 'if true then don\'t plot the results']
]

plot = True
try:
  matplotlib.use("GTK3Agg")
except ImportError:
  print("can't load matplotlib")
  plot = False
sns.set(style="darkgrid")


def run_bash(cmd: str):
  stream = os.popen(cmd)
  return stream.read()


def test_split_json():
  print(split_json("{}{qsdf}{sdfqfh}"))


def split_json(cmd_out):
  jsons = []
  n_accol = 0
  previous_split_char = 0
  for i, c in enumerate(cmd_out):
    if c == '{':
      n_accol += 1
    if c == '}':
      n_accol -= 1
    if n_accol == 0:
      # then split the json
      jsons.append(cmd_out[previous_split_char: i+1])
      previous_split_char = i+1
  return jsons


def parse_str(e):
  if e is None:
    return '0'
  return str(e)


def main():
  global plot
  args = argparse.parse_cmd(arguments)
  if args['no_plot']:
    plot = False
  server = args['server']
  remote_dir = args['remote_dir']
  out = run_bash(f'ssh {server} "cd {remote_dir} && cat */trial.json"')
  jsons = split_json(out)
  csv_content = 'experiment_id,factor,framework,depth,score\n'
  for trial in jsons:
    trial = json.loads(trial)
    values = trial['hyperparameters']['values']
    csv_values = [trial['trial_id'], values['factor'], values['framework'], values['depth'], trial['score']]
    csv_content += ','.join(list(map(parse_str, csv_values))) + '\n'

  with open(args['csv_path'], 'w') as f:
    f.write(csv_content)

  if not plot:
    return

  # plot the results
  data = pd.read_csv(args['csv_path'])
  print(data)
  f, ax = plt.subplots(figsize=(9, 6))
  data = data.drop_duplicates(subset=["factor", "depth"])
  data = data.pivot("factor", "depth", "score")
  print(data)
  sns.heatmap(data, annot=True, vmin=0, vmax=1, cmap='CMRmap')
  plt.show()


if __name__ == "__main__":
  # test_split_json()
  main()
