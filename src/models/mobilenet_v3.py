"""MobileNet v3 models for Keras."""
'''

  Reference:
  - [Searching for MobileNetV3](
      https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)

Code from 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/mobilenet_v3.py 
adapted to upstride

  The following table describes the performance of MobileNets:
  ------------------------------------------------------------------------
  MACs stands for Multiply Adds
  
  |Classification Checkpoint|MACs(M)|Parameters(M)|Top1 Accuracy|Pixel1 CPU(ms)|
  |---|---|---|---|---|
  | mobilenet_v3_large_1.0_224              | 217 | 5.4 |   75.6   |   51.2  |
  | mobilenet_v3_large_0.75_224             | 155 | 4.0 |   73.3   |   39.8  |
  | mobilenet_v3_large_minimalistic_1.0_224 | 209 | 3.9 |   72.3   |   44.1  |
  | mobilenet_v3_small_1.0_224              | 66  | 2.9 |   68.1   |   15.8  |
  | mobilenet_v3_small_0.75_224             | 44  | 2.4 |   65.4   |   12.8  |
  | mobilenet_v3_small_minimalistic_1.0_224 | 65  | 2.0 |   61.9   |   12.2  |

6.1.1    Training setup
We train our models using synchronous training setup on4x4 TPU Pod [24] using standard tensorflow 
RMSPropOp-timizer with 0.9 momentum. We use the initial learning rateof  0.1,
with  batch  size  4096  (128  images  per  chip),  andlearning  rate  decay  rate  of  0.01
every  3  epochs.   We  use dropout of 0.8, and l2 weight decay 1e-5 and the same image preprocessing as Inception [42].
Finally we use expo-nential moving average with decay 0.9999. '''

import tensorflow as tf 
from .generic_model import GenericModel

BATCH_NORM_MOMEMTUM = 0.9 
KERNEL_REGULARIZER = tf.keras.regularizers.l2(l2=1e-3) 

def _make_divisble(v, divisor=8, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v

def correct_pad(inputs, kernel_size):
  """Returns a tuple for zero-padding for 2D convolution with downsampling.
  Args:
      input_size: An integer or tuple/list of 2 integers.
      kernel_size: An integer or tuple/list of 2 integers.
  Returns:
      A tuple.
  """
  img_dim = 1
  if type(inputs) == list:
    inputs = inputs[0]
  is_channel_first = False
  if is_channel_first: # TODO If and else are the same. investigate this.
    input_size = tf.keras.backend.int_shape(inputs)[img_dim+1:(img_dim + 3)]
  else:
    input_size = tf.keras.backend.int_shape(inputs)[img_dim+1:(img_dim + 3)]

  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)

  if input_size[0] is None:
    adjust = (1, 1)
  else:
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

  correct = (kernel_size[0] // 2, kernel_size[1] // 2)

  return ((correct[0] - adjust[0], correct[0]),
          (correct[1] - adjust[1], correct[1]))

def relu(x):
  return tf.nn.relu(x)

def hard_sigmoid(x):
  return tf.nn.relu6(x + 3.) / 6.

def hard_swish(x):
  return x * hard_sigmoid(x) 

class _MobileNetV3(GenericModel):
  def __init__(self, *args, **kwargs):
    self.last_block_output_shape = 3
    self.channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    super().__init__(*args, **kwargs)

  def _inverted_res_block(self, expansion, filters, kernel_size, stride, se_ratio,
                          activation_fn, block_id):
    layers = self.layers()  # we don't want to switch between tf and upstride in this block
    inputs = self.x
    prefix = 'expanded_conv/'
    infilters = self.last_block_output_shape 
    if block_id:
      # Expand
      prefix = 'expanded_conv_{}/'.format(block_id)
      self.x = layers.Conv2D(
          _make_divisble(infilters * expansion),
          kernel_size=1,
          kernel_regularizer=KERNEL_REGULARIZER,
          padding='same',
          use_bias=False,
          name=prefix + 'expand')(
              self.x)
      self.x = layers.BatchNormalization(
          axis=self.channel_axis,
          epsilon=1e-3,
          momentum=BATCH_NORM_MOMEMTUM,
          name=prefix + 'expand/BatchNorm')(
              self.x)
      self.x = layers.Activation(activation_fn, name=prefix + '1a_' + activation_fn.__name__)(self.x)
    if stride == 2:
      self.x = layers.ZeroPadding2D(
          padding=correct_pad(self.x, kernel_size),
          name=prefix + 'depthwise/pad')(
              self.x)
    self.x = layers.DepthwiseConv2D(
        kernel_size,
        kernel_regularizer=KERNEL_REGULARIZER,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'depthwise')(
            self.x)
    self.x = layers.BatchNormalization(
        axis=self.channel_axis,
        epsilon=1e-3,
        momentum=BATCH_NORM_MOMEMTUM,
        name=prefix + 'depthwise/BatchNorm')(
            self.x)
    self.x = layers.Activation(activation_fn, name=prefix + activation_fn.__name__)(self.x)

    if se_ratio:
      self._se_block(_make_divisble(infilters * expansion), se_ratio, prefix)

    self.x = layers.Conv2D(
        filters,
        kernel_size=1,
        kernel_regularizer=KERNEL_REGULARIZER,
        padding='same',
        use_bias=False,
        name=prefix + 'project')(
            self.x)
    self.x = layers.BatchNormalization(
        axis=self.channel_axis,
        epsilon=1e-3,
        momentum=BATCH_NORM_MOMEMTUM,
        name=prefix + 'project/BatchNorm')(
            self.x)

    if stride == 1 and infilters == filters:
      self.x = layers.Add(name=prefix + 'Add')([inputs, self.x])

    self.last_block_output_shape = filters

  def _se_block(self, filters, se_ratio, prefix):
    inputs_seblock = self.x
    layers = self.layers()
    self.x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(self.x)
    self.x = layers.Dense(
        _make_divisble(filters * se_ratio),
        kernel_regularizer=KERNEL_REGULARIZER,
        name=prefix + 'squeeze_excite/Dense')(
            self.x)
    self.x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(self.x)
    self.x = layers.Dense(
        filters,
        kernel_regularizer=KERNEL_REGULARIZER,
        name=prefix + 'squeeze_excite/Dense_1')(
            self.x)
    self.x = layers.Activation(hard_sigmoid, name=prefix + 'squeeze_excite/Hard_Sigmoid')(self.x)
    self.x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs_seblock, self.x])

  def model(self):
    self.x = self.layers().Conv2D(
        16 // self.factor,
        kernel_size=3,
        kernel_regularizer=KERNEL_REGULARIZER,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(self.x)
    self.x = self.layers().BatchNormalization(
        axis=self.channel_axis, epsilon=1e-3,
        momentum=BATCH_NORM_MOMEMTUM, name='Conv/BatchNorm')(self.x)
    self.x = self.layers().Activation(hard_swish, name='Conv/Hard_Swish')(self.x)

    for i, layer in enumerate(self.config):
      self._inverted_res_block(
        layer[0], # expansion
        _make_divisble(layer[1] * self.alpha) // self.factor, # filters
        layer[2], # kernel
        layer[3], # stride
        layer[4], # se_ratio
        layer[5], # activation
        i  # block_id
        )
    
    if type(self.x) == list:
      last_conv_ch = _make_divisble(self.x[0].shape[self.channel_axis] * 6)
    else:
      last_conv_ch = _make_divisble(self.x.shape[self.channel_axis] * 6)


    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if self.alpha > 1.0: 
      self.last_point_ch = _make_divisble(self.last_point_ch * self.alpha)
    self.x = self.layers().Conv2D(
        last_conv_ch,
        kernel_size=1,
        kernel_regularizer=KERNEL_REGULARIZER,
        padding='same',
        use_bias=False,
        name='Conv_1')(self.x)
    self.x = self.layers().BatchNormalization(
        axis=self.channel_axis, epsilon=1e-3,
        momentum=BATCH_NORM_MOMEMTUM, name='Conv_1/BatchNorm')(self.x)
    self.x = self.layers().Activation(hard_swish, name='Conv_1/Hard_Swish')(self.x)
    
    self.x = self.layers().GlobalAveragePooling2D()(self.x)
    if self.channel_axis == 1:
      self.x = self.layers().Reshape((last_conv_ch, 1, 1))(self.x)
    else:
      self.x = self.layers().Reshape((1, 1, last_conv_ch))(self.x)
    if self.dropout_rate > 0:
      self.x = self.layers().Dropout(self.dropout_rate)(self.x)
    
    self.x = self.layers().Conv2D(
        self.last_point_ch // self.factor,
        kernel_size=1,
        kernel_regularizer=KERNEL_REGULARIZER,
        padding='same',
        use_bias=False,
        name='Conv_2')(self.x)
    self.x = self.layers().Activation(hard_swish, name='Conv_2/Hard_Swish')(self.x)

    self.x = self.layers().Flatten()(self.x)
    self.x = self.layers().Dense(
        self.label_dim, 
        use_bias=True,
        kernel_regularizer=KERNEL_REGULARIZER,
        name='Logits')(self.x)

  
class MobileNetV3Small(_MobileNetV3):
  """MobileNetV3 small version has 11 layers of inverted blocks.

  Args:
    alpha: controls the width of the network. This is known as the
      depth multiplier in the MobileNetV3 paper, but the name is kept for
      consistency with MobileNetV1 in Keras.
      - If `alpha` < 1.0, proportionally decreases the number
          of filters in each layer.
      - If `alpha` > 1.0, proportionally increases the number
          of filters in each layer.
      - If `alpha` = 1, default number of filters from the paper
          are used at each layer.
    dropout_rate: default rate is set to zero.
    config: configuration for the inverted residual blocks.
  """
  def __init__(self, *args, **kwargs):
    self.alpha = 1.0
    self.dropout_rate = 0
    self.config = [
    
    # TODO plan to replace activation with strings
    # expansion, filters, kernel, stride, se_ratio, activation
    # paper 0.25 at the first block, using make divisble makes filters to 8 rather 16 and causing dimension issue at Multiply
        (1,        16,      3,      2,      None,       relu), 
        (4.5,      24,      3,      2,      None,       relu), 
        (3.66,     24,      3,      1,      None,       relu), 
        (4,        40,      5,      2,      0.25,       hard_swish), 
        (6,        40,      5,      1,      0.25,       hard_swish), 
        (6,        40,      5,      1,      0.25,       hard_swish), 
        (3,        48,      5,      1,      0.25,       hard_swish), 
        (3,        48,      5,      1,      0.25,       hard_swish), 
        (6,        96,      5,      2,      0.25,       hard_swish), 
        (6,        96,      5,      1,      0.25,       hard_swish), 
        (6,        96,      5,      1,      0.25,       hard_swish), 
    ]
    self.last_point_ch = 1024
    super().__init__(*args, **kwargs)

class MobileNetV3Large(_MobileNetV3):
  """MobileNetV3 large version has 15 layers of inverted blocks.

  Args:
    alpha: controls the width of the network. This is known as the
      depth multiplier in the MobileNetV3 paper, but the name is kept for
      consistency with MobileNetV1 in Keras.
      - If `alpha` < 1.0, proportionally decreases the number
          of filters in each layer.
      - If `alpha` > 1.0, proportionally increases the number
          of filters in each layer.
      - If `alpha` = 1, default number of filters from the paper
          are used at each layer.
    dropout_rate: default rate is set to zero.
    config: configuration for the inverted residual blocks.
  """
  def __init__(self, *args, **kwargs):
    self.alpha = 1.0
    self.dropout_rate = 0
    self.config = [
    # expansion, filters, kernel, stride, se_ratio, activation   
        (1,        16,      3,      1,      None,       relu),
        (4,        24,      3,      2,      None,       relu),
        (3,        24,      3,      1,      None,       relu),
        (3,        40,      5,      2,      0.25,       relu),
        (3,        40,      5,      1,      0.25,       relu),
        (3,        40,      5,      1,      0.25,       relu),
        (6,        80,      3,      2,      None,       hard_swish),
        (2.5,      80,      3,      1,      None,       hard_swish),
        (2.3,      80,      3,      1,      None,       hard_swish),
        (2.3,      80,      3,      1,      None,       hard_swish),
        (6,        112,     3,      1,      0.25,       hard_swish),
        (6,        112,     3,      1,      0.25,       hard_swish),
        (6,        160,     5,      2,      0.25,       hard_swish),
        (6,        160,     5,      1,      0.25,       hard_swish),
        (6,        160,     5,      1,      0.25,       hard_swish),
    ]
    self.last_point_ch = 1280
    super().__init__(*args, **kwargs)