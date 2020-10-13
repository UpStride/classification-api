from typing import List
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


def gumbel_softmax(logits, gumble_noise=True):
  """please have a look at https://arxiv.org/pdf/1611.01144.pdf for gumble definition
  """
  global temperature
  if gumble_noise:
    u = tf.random.uniform(minval=0.0, maxval=1.0, shape=tf.shape(logits))
    noise = -tf.math.log(-tf.math.log(u))
  else:
    noise = 0.0001
  noisy_logits = (noise + logits) / temperature
  return tf.math.softmax(noisy_logits)


def get_mask(binary_vectors: List[tf.Tensor], g: List[float]):
  vectors = [g[i] * binary_vectors[i] for i in range(len(g))]
  vectors = tf.stack(vectors, axis=0)
  vector = tf.reduce_sum(vectors, axis=0)
  return vector


class ChannelMasking(tf.keras.layers.Layer):
  def __init__(self, min: int, max: int, step: int, name: str, gumble_noise=True):
    super().__init__(name=name)
    self.min = min
    self.max = max
    self.step = step
    self.channel_sizes = []
    self.gumble_noise = gumble_noise
    for i in range(self.min, self.max+1, self.step):
      self.channel_sizes.append(i)

  def build(self, input_shape):
    self.alpha = self.add_weight(name=f"alpha",
                                 shape=(len(self.channel_sizes),),
                                 initializer=tf.keras.initializers.Constant(value=1.))
    self.binary_vectors = create_binary_vector(self.channel_sizes, dtype=self.alpha.dtype)

  def call(self, inputs):
    self.g = gumbel_softmax(self.alpha, self.gumble_noise)
    mask = get_mask(self.binary_vectors,  self.g)

    # work with channel last but not channel first
    if tf.keras.backend.image_data_format() == 'channels_first':
      mask = tf.reshape(mask, [1, self.channel_sizes[-1], 1, 1])
    if type(inputs) == list:
      return [mask * inputs[i] for i in range(len(inputs))]
    else:
      return mask * inputs
