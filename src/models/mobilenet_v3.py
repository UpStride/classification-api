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
'''
import tensorflow as tf 
from .generic_model import GenericModel

"""
  Arguments:
    input_shape: Optional shape tuple, to be specified if you would
      like to use a model with an input image resolution that is not
      (224, 224, 3).
      It should have exactly 3 inputs channels (224, 224, 3).
      You can also omit this option if you would like
      to infer input_shape from an input_tensor.
      If you choose to include both input_tensor and input_shape then
      input_shape will be used if they match, if the shapes
      do not match then we will throw an error.
      E.g. `(160, 160, 3)` would be one valid value.
    alpha: controls the width of the network. This is known as the
      depth multiplier in the MobileNetV3 paper, but the name is kept for
      consistency with MobileNetV1 in Keras.
      - If `alpha` < 1.0, proportionally decreases the number
          of filters in each layer.
      - If `alpha` > 1.0, proportionally increases the number
          of filters in each layer.
      - If `alpha` = 1, default number of filters from the paper
          are used at each layer.
    minimalistic: In addition to large and small models this module also
      contains so-called minimalistic models, these models have the same
      per-layer dimensions characteristic as MobilenetV3 however, they don't
      utilize any of the advanced blocks (squeeze-and-excite units, hard-swish,
      and 5x5 convolutions). While these models are less efficient on CPU, they
      are much more performant on GPU/DSP.
    include_top: Boolean, whether to include the fully-connected
      layer at the top of the network. Defaults to `True`.
    weights: String, one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: Optional Keras tensor (i.e. output of
      `layers.Input()`)
      to use as image input for the model.
    pooling: String, optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model
          will be the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a
          2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: Integer, optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    dropout_rate: fraction of the input units to drop on the last layer.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape or invalid alpha, rows when
      weights='imagenet'
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
"""

BATCH_NORM_MOMEMTUM = 0.9

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
  is_channel_fist = False
  if is_channel_fist: # TODO If and else are the same. investigate on this.
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
  return tf.nn.relu6(x)

def hard_sigmoid(x):
  return tf.nn.relu6((x + 3.) * (1. / 6.))

def hard_swish(x):
  return x * hard_sigmoid(x) 

tf.keras.activations.hard_sigmoid

class _MobileNetV3(GenericModel):
  def __init__(self, *args, **kwargs):
    self.last_block_output_shape = 3
    self.channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    super().__init__(*args, **kwargs)

  def _inverted_res_block(self, expansion, filters, kernel_size, stride, se_ratio,
                          activation_fn, block_id):
    layers = self.layers()  # we don't want to switch between tf and upstride in this block
    # shouldn't this be deepcopy??
    inputs = self.x
    prefix = 'expanded_conv/'
    infilters = self.last_block_output_shape 
    if block_id:
      # Expand
      prefix = 'expanded_conv_{}/'.format(block_id)
      self.x = layers.Conv2D(
          _make_divisble(infilters * expansion),
          kernel_size=1,
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
      self.x = layers.Activation(activation_fn)(self.x)
    if stride == 2:
      self.x = layers.ZeroPadding2D(
          padding=correct_pad(self.x, kernel_size),
          name=prefix + 'depthwise/pad')(
              self.x)
    self.x = layers.DepthwiseConv2D(
        kernel_size,
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
    self.x = layers.Activation(activation_fn)(self.x)
    if se_ratio:
      self._se_block(_make_divisble(infilters * expansion), se_ratio, prefix)

    self.x = layers.Conv2D(
        filters,
        kernel_size=1,
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
    inputs = self.x
    layers = self.layers()
    self.x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(self.x)
    if tf.keras.backend.image_data_format() == 'channels_first':
      self.x = layers.Reshape((filters, 1, 1))(self.x)
    else:
      self.x = layers.Reshape((1, 1, filters))(self.x)
    self.x = layers.Conv2D(
        _make_divisble(filters * se_ratio),
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv')(
            self.x)
    self.x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(self.x)
    self.x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv_1')(
            self.x)
    self.x = layers.Activation(hard_sigmoid)(self.x)
    self.x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, self.x])

  def model(self):
    self.x = self.layers().Conv2D(
        16 // self.factor,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(self.x)
    self.x = self.layers().BatchNormalization(
        axis=self.channel_axis, epsilon=1e-3,
        momentum=BATCH_NORM_MOMEMTUM, name='Conv/BatchNorm')(self.x)
    self.x = self.layers().Activation(hard_swish, name='hard_swish')(self.x)

    for layer in self.config:
      self._inverted_res_block(
        layer[0], # expansion
        _make_divisble(layer[1] * self.alpha) // self.factor, # filters
        layer[2], # kernel
        layer[3], # stride
        layer[4], # se_ration
        layer[5], # activation
        layer[6]  # block_id
        )
    
    if type(self.x) == list:
      last_conv_ch = _make_divisble(self.x[0].shape[self.channel_axis] * 6)
    else:
      last_conv_ch = _make_divisble(self.x.shape[self.channel_axis] * 6)


    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if self.alpha > 1.0: 
      self.last_point_ch = _make_divisble(self.last_point_ch * self.alpha) // self.factor
    self.x = self.layers().Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_1')(self.x)
    self.x = self.layers().BatchNormalization(
        axis=self.channel_axis, epsilon=1e-3,
        momentum=BATCH_NORM_MOMEMTUM, name='Conv_1/BatchNorm')(self.x)
    self.x = self.layers().Activation(hard_swish)(self.x)
    
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
        padding='same',
        use_bias=True,
        name='Conv_2')(self.x)
    self.x = self.layers().Activation(hard_swish)(self.x)

    self.x = self.layers().Conv2D(self.label_dim, kernel_size=1, padding='same', name='Logits')(self.x)
    self.x = self.layers().Flatten()(self.x)

  
class MobileNetV3Small(_MobileNetV3):
  def __init__(self, *args, **kwargs):
    self.alpha = 1.0
    self.dropout_rate = 0
    self.config = [
    # expansion, filters, kernel, stride, se_ratio, activation, block_id
        (1,        16,      3,      2,      None,       relu,      0), 
        (4.5,      24,      3,      2,      None,       relu,      1), 
        (3.66,     24,      3,      1,      None,       relu,      2), 
        (4,        40,      3,      2,      None,       relu,      3), 
        (6,        40,      3,      1,      None,       relu,      4), 
        (6,        40,      3,      1,      None,       relu,      5), 
        (3,        48,      3,      1,      None,       relu,      6), 
        (3,        48,      3,      1,      None,       relu,      7), 
        (6,        96,      3,      2,      None,       relu,      8), 
        (6,        96,      3,      1,      None,       relu,      9), 
        (6,        96,      3,      1,      None,       relu,      10), 
    ]
    self.last_point_ch = 1024
    super().__init__(*args, **kwargs)

class MobileNetV3Large(_MobileNetV3):
  def __init__(self, *args, **kwargs):
    self.alpha = 1.0
    self.dropout_rate = 0
    self.config = [
    # expansion, filters, kernel, stride, se_ratio, activation,       block_id
        (1,        16,      3,      1,      None,       relu,            0), 
        (4,        24,      3,      2,      None,       relu,            1), 
        (3,        24,      3,      1,      None,       relu,            2), 
        (3,        40,      5,      2,      0.25,       relu,            3), 
        (3,        40,      5,      1,      0.25,       relu,            4), 
        (3,        40,      5,      1,      0.25,       relu,            5), 
        (6,        80,      3,      2,      None,       hard_swish,      6), 
        (2.5,      80,      3,      1,      None,       hard_swish,      7), 
        (2.3,      80,      3,      1,      None,       hard_swish,      8), 
        (2.3,      80,      3,      1,      None,       hard_swish,      9), 
        (6,        112,     3,      1,      0.25,       hard_swish,      10), 
        (6,        112,     3,      1,      0.25,       hard_swish,      11), 
        (6,        160,     5,      2,      0.25,       hard_swish,      12), 
        (6,        160,     5,      1,      0.25,       hard_swish,      13), 
        (6,        160,     5,      1,      0.25,       hard_swish,      14), 
    ]
    self.last_point_ch = 1280
    super().__init__(*args, **kwargs)