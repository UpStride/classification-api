"""Script to benchmark several versions of Tensorflow and Upstride Tech on different hardware and docker platforms

to start a new benchmark, you can run 
python inference_benchmark.py --yaml_config conf1.yml conf2.yml --comments "small test"
"""
import os
from typing import List
import requests
import json
import yaml
import upstride_argparse as argparse

inference_arguments = [
    [int, "batch_size", 1, 'The size of batch per gpu', lambda x: x > 0],
    [str, "comments", "", 'some comment about this benchmark run. Will be displayed on the model zoo'],
    [bool, 'cpu', False, 'is True then force cpu use'],
    [int, 'cuda_visible_device', 0, 'the gpu to run the benchmark on'],
    ['list[str]', "docker_images", [], "list of docker images to test"],
    ['list[str]', "engines", [], "list of engines to test"],
    [str, "model_path", "", 'Specify the model path, to work on a real model instead of a fake one with random weights'],
    ['list[str]', "models", [], "list of models to test"],
    [str, "output", "results.md", "file with results"],
    [str, "profiling_dir", "/tmp", "dir where profiling files will be written"],
    [int, 'n_steps', 10, "number of steps to run the inference. The higher the better"],
    [bool, "tensorrt", False, "if true then models will be converted to tensorrt"],
    [str, "tensorrt_precision", 'FP32', 'Provide precision FP32 or FP16 for optimizing tensorrt'],
    ['list[str]', "yaml_config", [], "config files there can be as many implemented these options"],
    [bool, "xla", False, "if true then use xla"],
    ['namespace', 'user', [
        [str, "email", "", "user email for the upstride platform"],
        [str, "password", "", "password for the upstride platform"],
    ]]
]


def create_all_environment_configs(conf):
  """Create a list of dict with the docker, model and engine to benchmark. 
  will create all possible triplet

  Returns:
      List of Dict with docker, model and engine
  """
  env_configs = []
  for docker in conf["docker_images"]:
    for model in conf["models"]:
      for engine in conf["engines"]:
        env_configs.append({"docker": docker,
                            "model": model,
                            "engine": engine})
  return env_configs


def prepare_docker(docker_images: List[str]):
  """download all docker images to prepare benchmark

  Args:
      docker_images (List[str]): should be formated as ["docker_tag:docker_label", ...]
  """
  for docker_image in docker_images:
    if docker_image == "local":
      continue
    print(f"Pulling {docker_image}")
    stream = os.popen(f"docker pull {docker_image}")
    out = stream.read()
    print(out)


def docker_run_cmd(docker, engine, model, conf):
  if engine == "tensorflow":
    factor = 1
  elif 'upstride' in engine:
    factor = int(engine.split('f')[-1])
    engine = engine[:-3]
  # dev note: all option need to have a space at the end
  python_cmd = f"python3 src/inference_benchmark.py "\
      f"--batch_size {conf['batch_size']} "\
      f"--engine {engine} "\
      f"--factor {factor} "\
      f"--model_name {model} "\
      f"--n_steps {conf['n_steps']} "\
      f"--profiler_path {conf['profiling_dir']} "
  if conf['tensorrt']:
    python_cmd += f"--export_tensorrt {conf['tensorrt']} "
    python_cmd += f"--tensorrt_precision {conf['tensorrt_precision']} "
  if conf['model_path']:
    python_cmd += f"--model_path {conf['model_path']} "
  if conf['xla']:
    python_cmd += f"--xla {conf['xla']} "

  if docker == "local":
    # then run without docker
    return python_cmd

  runtime = f"--gpus all -e CUDA_VISIBLE_DEVICES={conf['cuda_visible_device']}" if "gpu" in docker and not conf['cpu'] else ""
  volumes = " -v $(pwd)/src:/src -v /tmp/docker:/tmp"
  # In the special case of tensorflow 2.2, add a volume to save the profiling
  # docker need also to be run with the privileged parameter to access gpu information
  if "tf2.2" in docker:
    volumes += " -v $(pwd)/profiling:/profiling --privileged=true"
  return f"docker run -it --rm {runtime} {volumes} {docker} {python_cmd}"


def format_results(env_configs, results, output_file):
  with open(output_file, "w") as f:
    f.write(f"| docker                                                       |engine      |model    |n_iteration|total time|time per iteration|FPS    |\n")
    f.write(f"|:------------------------------------------------------------:|:----------:|:-------:|:---------:|:--------:|:----------------:|:-----:|\n")
    for i in range(len(results)):
      result = results[i]
      env_config = env_configs[i]
      time_per_iteration = result['total_time']/result['n_iterations']
      fps = 1/time_per_iteration
      line = f"| {env_config['docker']} | {env_config['engine']: <10} | {env_config['model']} | {result['n_iterations']} | {result['total_time']:.2f} | {time_per_iteration:.3f} | {fps:.1f} |\n"
      f.write(line)


def call_api(result, env_config, conf):
  results = {
      "batch_size": conf["batch_size"],
      "comments": conf["comments"],
      "docker_image": env_config['docker'],
      "engine": env_config['engine'],
      'hardware': result['gpu'],
      "model": env_config['model'],
      "n_iterations": result["n_iterations"],
      "n_params": result["n_params"],
      'tensorrt': conf['tensorrt'],
      'tensorrt_precision': conf['tensorrt_precision'],
      'xla': conf['xla'],
      "total_time": result["total_time"],
  }
  r = requests.post('https://api.upstride.io/inference_benchmark', json={'result': results}, auth=(conf['user']['email'], conf['user']['password']))
  # r = requests.post('http://127.0.0.1:8888/inference_benchmark', json={'result': results}, auth=(conf['user']['email'], conf['user']['password']))
  print(r.status_code)
  if r.status_code == 400:
    print(r.content)
    raise Exception("issue connecting backend")


def main():
  conf = argparse.parse_cmd(inference_arguments)
  print(conf)
  # currently first gpu is being picked if there are multiple GPUs.
  # conf['hardware']['gpu'] = get_gpu_info().get('name')[0]
  prepare_docker(conf["docker_images"])
  # benchmark all docker images against all models against all engine
  env_configs = create_all_environment_configs(conf)
  results = []
  for env_config in env_configs:
    print(f"Benchmark {env_config['model']} using {env_config['engine']} on {env_config['docker']}")
    cmd = docker_run_cmd(env_config['docker'], env_config['engine'], env_config['model'], conf)
    print(cmd)
    stream = os.popen(cmd)
    out = stream.read()
    print(out)
    # look for a correct output
    i = 2
    while out.split('\n')[-i][0] != '{':
      i += 1
      # print(i)
    r = json.loads(out.split('\n')[-i])
    call_api(r, env_config, conf)
    results.append(r)
  format_results(env_configs, results, conf["output"])


if __name__ == "__main__":
  main()
