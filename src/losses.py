from src.models.fbnetv2 import ChannelMasking
from submodules.global_dl.training import metrics
import tensorflow as tf

def _count_parameters_conv2d(layer):
  if type(layer.input_shape) is list:
    input_shape = layer.input_shape[0]
  else:
    input_shape = layer.input_shape

  if type(layer.output_shape) is list:
    output_shape = layer.output_shape[0]
  else:
    output_shape = layer.output_shape

  if layer.data_format == "channels_first":
    input_channels = layer.input_shape[1]
    output_channels, h, w, = output_shape[1:]
  elif layer.data_format == "channels_last":
    input_channels = input_shape[3]
    h, w, output_channels = output_shape[1:]
  w_h, w_w = layer.kernel_size

  num_params = output_channels * input_channels * w_h * w_w

  if layer.use_bias:
    num_params += output_channels

  return int(num_params)


def flops_loss(model):
  """loss function defined by number of flops, usefull for Differential Architecture Search

  This function is compatible both with TensorFlow and UpStride engine
  
  Args:
      model: Keras model containing some ChannelMasking layers
  
  Returns:
      float: loss
  """
  loss = 0
  for layer in model.layers:
    if "Conv2D" in str(type(layer)) and "Depthwise" not in str(type(layer)):
      flops = metrics._count_flops_conv2d(layer)
    if type(layer) == ChannelMasking:
      # flops is the number of flops of the channel just before ChannelMasking
      g = layer.g
      param_ratio = [flops * (layer.min + i * layer.step)/layer.max for i in range(layer.g.shape[0])]
      loss += tf.math.reduce_sum(g * tf.convert_to_tensor(param_ratio))
  return loss


def parameters_loss(model):
  loss = 0
  for layer in model.layers:
    if "Conv2D" in str(type(layer)) and "Depthwise" not in str(type(layer)):
      n_params = _count_parameters_conv2d(layer)
    if type(layer) == ChannelMasking:
      # parameters are the number of parameters of the channel just before ChannelMasking
      g = layer.g
      param_ratio = [n_params * (layer.min + i * layer.step)/layer.max for i in range(g.shape[0])]
      loss += tf.math.reduce_sum(g * tf.convert_to_tensor(param_ratio))
  return loss
