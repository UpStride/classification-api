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
    [str, 'local_dir', '', 'path of the directory containing the results']
]

try:
  matplotlib.use("GTK3Agg")
except ImportError:
  pass
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


def main():
  args = argparse.parse_cmd(arguments)
  server = args['server']
  remote_dir = args['remote_dir']
  out = run_bash(f'ssh {server} "cd {remote_dir} && cat */trial.json"')
  jsons = split_json(out)
  for trial in jsons:
    trial = json.loads(trial)
    print()
    print(trial['trial_id'])
    print(trial['hyperparameters']['values'])
    print(trial['score'])



if __name__ == "__main__":
  # test_split_json()
  main()
