import json
import os
from typing import List
import yaml
import tensorflow as tf
import numpy as np

# global variables needed for softmax gumbel computation in MaskConv
temperature = 5.0  # should be multiply by 0.956 at the end of every epoch, see section 4.1 in the paper


def define_temperature(new_temperature):
  global temperature
  temperature = new_temperature


def create_binary_vector(channel_sizes: List[int], dtype) -> List[tf.Tensor]:
  """this function return a list of vector with ones at the beginning and zeros at the end
  it uses numpy because there is no reason for these operations to be inside the tensorflow graph.

  Args:
      channel_sizes (List[int]): number of channels in the convolution

  Returns:
      List[tf.Tensor]: list of vector like [1., 1., 1., 0., 0., 0.]
  """
  binary_vectors = []
  max_size = channel_sizes[-1]
  for i in range(len(channel_sizes)):
    ones = np.ones(channel_sizes[i])
    zeros = np.zeros(max_size - channel_sizes[i])
    binary_vectors.append(tf.convert_to_tensor(np.concatenate([ones, zeros], 0), dtype=dtype))
  return binary_vectors


def gumbel_softmax(logits, gumble_noise=False):
  """please have a look at https://arxiv.org/pdf/1611.01144.pdf for gumble definition
  """
  global temperature

  if gumble_noise:
    # Gumble distribution -log(-log(u)), where u ~ (0,1) is a uniform distribution and
    # must be sampled from the open-interval `(0, 1)` but tf.random.uniform generates samples
    #  where The lower bound minval is included in the range like [0, 1). To make sure the range
    # to be (0, 1), np.finfo(float).tiny is used as minval which gives a tiny postive floating point number
    u = tf.random.uniform(minval=np.finfo(float).tiny, maxval=1.0, shape=tf.shape(logits))
    noise = -tf.math.log(-tf.math.log(u))  # Noise from gumbel distribution
  else:
    noise = 0.0001
  # During mixed precision training, Weight Variable data type is inferred from "inputs" during call method
  # This makes alpha to be converted to float16. 
  # Since we are computing softmax at the end, we need to convert logits(alpha) to float32
  logits = tf.cast(logits, tf.float32) 
  noisy_logits = (noise + logits) / temperature

  return tf.math.softmax(noisy_logits)


def get_mask(binary_vectors: List[tf.Tensor], g: List[float]):
  vectors = [g[i] * binary_vectors[i] for i in range(g.shape[0])]
  vectors = tf.stack(vectors, axis=0)
  vector = tf.reduce_sum(vectors, axis=0)
  return vector


class ChannelMasking(tf.keras.layers.Layer):
  def __init__(self, min: int, max: int, step: int, name: str, gumble_noise=True, regularizer=None):
    super().__init__(name=name)
    self.min = min
    self.max = max
    self.step = step
    self.channel_sizes = []
    self.gumble_noise = gumble_noise
    self.regularizer = regularizer
    for i in range(self.min, self.max+1, self.step):
      self.channel_sizes.append(i)

  def build(self, input_shape):
    self.alpha = self.add_weight(name=f"alpha",
                                 shape=(len(self.channel_sizes),),
                                 initializer=tf.keras.initializers.Constant(value=1.), regularizer=self.regularizer)
    self.binary_vectors = create_binary_vector(self.channel_sizes, dtype=self.alpha.dtype)

  def call(self, inputs):
    self.g = gumbel_softmax(self.alpha, self.gumble_noise)
    mask = get_mask(self.binary_vectors,  self.g)
    # Convert mast from Float32 to Float16 during mixed precision. 
    mask = tf.cast(mask, dtype=inputs.dtype)

    # work with channel last but not channel first
    if tf.keras.backend.image_data_format() == 'channels_first':
      mask = tf.reshape(mask, [1, self.channel_sizes[-1], 1, 1])
    if type(inputs) == list:
      return [mask * inputs[i] for i in range(len(inputs))]
    else:
      return mask * inputs


def exponential_decay(initial_value, decay_steps, decay_rate):
  """
          Applies exponential decay to initial value
       Args:
          initial_value: The initial learning value
          decay_steps: Number of steps to decay over
          decay_rate: decay rate
      """
  return lambda step: initial_value * decay_rate ** (step / decay_steps)


def split_trainable_weights(model, arch_params_name='alpha'):
  """
      split the model parameters  in weights and architectural params
  """
  weights = []
  arch_params = []
  for trainable_weight in model.trainable_variables:
    if arch_params_name in trainable_weight.name:
      arch_params.append(trainable_weight)
    else:
      weights.append(trainable_weight)
  if not arch_params:
    raise ValueError(f"No architecture parameters found by the name {arch_params_name}")
  return weights, arch_params


def post_training_analysis(model, saved_file_path):
  layer_name = ''
  saved_file_content = {}
  for layer in model.layers:
    # if type(layer) == tf.keras.Conv2D:
    #     layer_name = layer.name
    if type(layer) == ChannelMasking and layer.name[-8:] == '_savable':
      layer_name = layer.name[:-8]
      max_alpha_id = int(tf.math.argmax(layer.alpha).numpy())
      value = layer.min + max_alpha_id * layer.step
      saved_file_content[layer_name] = value
  print(saved_file_content)
  with open(saved_file_path, 'w') as f:
    yaml.dump(saved_file_content, f)


def save_arch_params(model, epoch, log_dir):
  json_file_path = os.path.join(log_dir, f'alpha.json')
  content = {}
  if os.path.exists(json_file_path):
    with open(json_file_path) as f:
      content = json.load(f)
  for layer in model.layers:
    if type(layer) == ChannelMasking:
      # need to convert from numpy.float32 to pure python float32 to prepare the dumps
      if str(epoch) not in content:
        content[str(epoch)] = {}
      content[str(epoch)][layer.name] = list(map(float, layer.alpha.numpy()))
  with open(json_file_path, 'w') as f:
    f.write(json.dumps(content))
