import tensorflow as tf
from .generic_model import GenericModel
import tensorflow.keras as tfk
from tensorflow.keras import activations, Sequential
from collections import namedtuple

drop_path_prob = None
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PDARTS = Genotype(normal=[('skip_connect', 0),
                          ('dil_conv_3x3', 1),
                          ('skip_connect', 0),
                          ('sep_conv_3x3', 1),
                          ('sep_conv_3x3', 1),
                          ('sep_conv_3x3', 3),
                          ('sep_conv_3x3', 0),
                          ('dil_conv_5x5', 4)],
                  normal_concat=range(2, 6),
                  reduce=[('avg_pool_3x3', 0),
                          ('sep_conv_5x5', 1),
                          ('sep_conv_3x3', 0),
                          ('dil_conv_5x5', 2),
                          ('max_pool_3x3', 0),
                          ('dil_conv_3x3', 1),
                          ('dil_conv_3x3', 1),
                          ('dil_conv_5x5', 3)],
                  reduce_concat=range(2, 6))

OPS = {
    'none': lambda layers, C, strides, trainable: Zero(strides),
    'avg_pool_3x3': lambda layers, C, strides, trainable: AvgPool(layers, strides),
    'max_pool_3x3': lambda layers, C, strides, trainable: MaxPool(layers, strides),
    'skip_connect': lambda layers, C, strides, trainable: Identity() if strides == 1 else FactorizedReduce(layers, C, trainable=trainable),
    'sep_conv_3x3': lambda layers, C, strides, trainable: SepConv(layers, C, C, 3, strides, 1, trainable=trainable),
    'sep_conv_5x5': lambda layers, C, strides, trainable: SepConv(layers, C, C, 5, strides, 2, trainable=trainable),
    'sep_conv_7x7': lambda layers, C, strides, trainable: SepConv(layers, C, C, 7, strides, 3, trainable=trainable),
    'dil_conv_3x3': lambda layers, C, strides, trainable: DilConv(layers, C, 3, strides, 2, 2, trainable=trainable),
    'dil_conv_5x5': lambda layers, C, strides, trainable: DilConv(layers, C, 5, strides, 4, 2, trainable=trainable),
    'conv_7x1_1x7': lambda layers, C, strides, trainable: Conv_7x1_1x7(layers, C, strides, trainable),
}


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1. - drop_prob
    shape = [tf.shape(x)[0], 1, 1, 1]
    mask_dist = tf.random.uniform(shape)
    mask = tf.math.floor(mask_dist + keep_prob)  # This line emulates tf.keras.backend.random_bernoulli
    x /= keep_prob
    x *= mask
  return x


class DataFormatHandler(tf.keras.layers.Layer):
  """ This class is needed because some Keras layer don't look at tf.keras.backend.image_data_format
  to know if there are channels first or last (for instance Batch Norm....)
  """

  def __init__(self):
    super().__init__()
    self.is_channels_first = True if tf.keras.backend.image_data_format() == 'channels_first' else False
    self.axis = 1 if self.is_channels_first else -1


# Primitives functions of P-Darts
class Identity(DataFormatHandler):
  def __init__(self):
    super().__init__()

  def call(self, x):
    return x


class Zero(DataFormatHandler):  # TODO
  def __init__(self, strides):
    super().__init__()
    self.strides = strides

  def call(self, x):
    if self.is_channels_first:
      n, c, h, w = x.shape
    else:
      n, h, w, c = x.shape
    h //= self.strides
    w //= self.strides
    padding = tf.zeros([n, c, h, w]) if self.is_channels_first else tf.zeros([n, h, w, c])
    return padding


class AvgPool(DataFormatHandler):
  def __init__(self, layers, strides):
    super().__init__()
    self.avgpool = layers.AveragePooling2D(3, strides=strides, padding='SAME')

  def call(self, x):
    x = self.avgpool(x)
    return x


class MaxPool(DataFormatHandler):
  def __init__(self, layers, strides):
    super().__init__()
    # self.pad = tf.keras.layers.ZeroPadding2D(padding=1)
    self.maxpool = layers.MaxPool2D(3, strides=strides, padding='SAME')

  def call(self, x):
    # x = self.pad(x)
    x = self.maxpool(x)
    return x


class SepConv(DataFormatHandler):  # TODO
  def __init__(self, layers, C_in, C_out, kernel_size, strides, padding, trainable=True):
    super().__init__()
    self.C_in = C_in
    self.C_out = C_out
    self.kernel_size = kernel_size
    self.stride = strides
    self.padding = padding
    self.affine = trainable

    self.pad = tf.keras.layers.ZeroPadding2D(padding)
    self.bn = layers.BatchNormalization(axis=self.axis, trainable=trainable)
    self.relu = layers.ReLU()
    self.dw_conv1 = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='VALID', depthwise_initializer='he_uniform', use_bias=False)
    self.conv1 = layers.Conv2D(C_in, kernel_size=1, padding='VALID', kernel_initializer='he_uniform', use_bias=False)
    self.dw_conv2 = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=1, padding='VALID', depthwise_initializer='he_uniform', use_bias=False)
    self.conv2 = layers.Conv2D(C_out, kernel_size=1, padding='VALID', kernel_initializer='he_uniform', use_bias=False)

  def call(self, x):
    x = self.relu(x)
    x = self.pad(x)
    x = self.dw_conv1(x)
    x = self.conv1(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.pad(x)
    x = self.dw_conv2(x)
    x = self.conv2(x)
    x = self.bn(x)
    return x


class Conv_7x1_1x7(DataFormatHandler):  # TODO
  def __init__(self, layers, C, strides, trainable):
    super().__init__()
    self.relu1 = layers.ReLU()
    self.pad1 = tf.keras.layers.ZeroPadding2D((0, 3))
    self.conv1 = layers.Conv2D(C, (1, 7), strides=(1, strides), padding='VALID', kernel_initializer='he_uniform', use_bias=False)
    self.pad2 = tf.keras.layers.ZeroPadding2D((3, 0))
    self.conv2 = layers.Conv2D(C, (7, 1), strides=(strides, 1), padding='VALID', kernel_initializer='he_uniform', use_bias=False)
    self.bn = layers.BatchNormalization(axis=self.axis, trainable=trainable)

  def call(self, x):
    x = self.relu1(x)
    x = self.pad1(x)
    x = self.conv1(x)
    x = self.pad2(x)
    x = self.conv2(x)
    x = self.bn(x)
    return x


class DilConv(DataFormatHandler):
  def __init__(self, layers, C_out, kernel_size, strides, padding, dilation, trainable=True):
    super().__init__()
    self.C_out = C_out
    self.kernel_size = kernel_size
    self.stride = strides
    self.padding = padding
    self.affine = trainable
    if strides != 1 and dilation != 1:
      if strides != dilation:
        raise ValueError("TensorFlow is not able to handle convolutions when both stride and dilation_rate are different than 1. A workaround has been implemented for the cases when strides == dilation_rate.")
      # If both strides and dilation are equal between them and they are different than 1,
      # then it is the equivalent of extracting a subgrid from the input and applying a convolution with both strides and dilation equal to 1
      # A representation can be visualized using the following tool: https://ezyang.github.io/convolution-visualizer/index.html
      self.do_workaround = True
    else:
      self.do_workaround = False

    self.relu = layers.ReLU()
    self.pad = tf.keras.layers.ZeroPadding2D(padding)
    if self.do_workaround:
      self.dw_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=1, padding='VALID', dilation_rate=1, depthwise_initializer='he_uniform', use_bias=False)
    else:
      self.dw_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='VALID', dilation_rate=dilation, depthwise_initializer='he_uniform', use_bias=False)
    self.conv = layers.Conv2D(C_out, kernel_size=1, padding='VALID', kernel_initializer='he_uniform', use_bias=False)
    self.bn = layers.BatchNormalization(axis=self.axis, trainable=trainable)

  def call(self, x):
    x = self.relu(x)
    x = self.pad(x)
    if self.do_workaround:
      x = x[:, :, ::self.stride, ::self.stride] if self.is_channels_first else x[:, ::self.stride, ::self.stride, :]
    x = self.dw_conv(x)
    x = self.conv(x)
    x = self.bn(x)
    return x


class FactorizedReduce(DataFormatHandler):
  def __init__(self, layers, C_out, trainable=True):
    super().__init__()
    assert C_out % 2 == 0
    self.relu = layers.ReLU()
    self.conv_1 = layers.Conv2D(C_out // 2, 1, strides=2, padding='VALID', kernel_initializer='he_uniform', use_bias=False)
    self.conv_2 = layers.Conv2D(C_out // 2, 1, strides=2, padding='VALID', kernel_initializer='he_uniform', use_bias=False)
    self.bn = layers.BatchNormalization(axis=self.axis, trainable=trainable)

  def call(self, x):
    x = self.relu(x)
    to_be_concat = [self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])] if self.is_channels_first else [self.conv_1(x), self.conv_2(x[:, 1:, 1:, :])]
    out = tf.concat(values=to_be_concat, axis=self.axis)
    out = self.bn(out)
    return out


class ReLUConvBN(DataFormatHandler):
  def __init__(self, layers, C_out, kernel_size, strides, padding, trainable=True):
    super().__init__()
    self.relu = layers.ReLU()
    self.pad = tf.keras.layers.ZeroPadding2D(padding)
    self.conv2d = layers.Conv2D(C_out, kernel_size, strides=strides, padding='VALID', kernel_initializer='he_uniform', use_bias=False)
    self.bn = layers.BatchNormalization(axis=self.axis, trainable=trainable)

  def call(self, x):
    x = self.relu(x)
    x = self.conv2d(x)
    x = self.bn(x)
    return x


class Cell(DataFormatHandler):
  def __init__(self, layers, genotype, C, reduction, reduction_prev):
    """
    Implementation note: 
    - The pytorch code has parameters C_prev_prev and C_prev. Here we don't need them as keras auto-infer
    input shape
    """
    super().__init__()
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(layers, C)
    else:
      self.preprocess0 = ReLUConvBN(layers, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(layers, C, 1, 1, 0)
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(layers, C, op_names, indices, concat, reduction)

  def _compile(self, layers, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = []
    for name, index in zip(op_names, indices):
      strides = 2 if reduction and index < 2 else 1
      op = OPS[name](layers, C, strides, True)
      self._ops += [op]
    self._indices = indices

  def call(self, s0, s1, drop_prob, training=False):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      old_h1 = states[self._indices[2*i]]
      old_h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(old_h1)
      h2 = op2(old_h2)
      # if training and drop_prob > 0.: # TODO investigate here
      if not isinstance(op1, Identity):
        h1 = drop_path(h1, drop_prob)
      if not isinstance(op2, Identity):
        h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]

    return tf.concat([states[i] for i in self._concat], axis=self.axis)


def cell(layers, genotype, C, reduction, reduction_prev, s0, s1, drop_prob, training=False):
  # Get info from genotype
  if reduction:
    op_names, indices = zip(*genotype.reduce)
    concat = genotype.reduce_concat
  else:
    op_names, indices = zip(*genotype.normal)
    concat = genotype.normal_concat

  ops = []
  for name, index in zip(op_names, indices):
    strides = 2 if reduction and index < 2 else 1
    op = OPS[name](layers, C, strides, True)
    ops += [op]

  s0 = FactorizedReduce(layers, C)(s0) if reduction_prev else ReLUConvBN(layers, C, 1, 1, 0)(s0)
  s1 = ReLUConvBN(layers, C, 1, 1, 0)(s1)
  states = [s0, s1]
  for i in range(len(op_names) // 2):
    old_h1 = states[indices[2*i]]
    old_h2 = states[indices[2*i+1]]
    op1 = ops[2*i]
    op2 = ops[2*i+1]
    h1 = op1(old_h1)
    h2 = op2(old_h2)
    # if training and drop_prob > 0.: # TODO investigate here
    if not isinstance(op1, Identity):
      h1 = drop_path(h1, drop_prob)
    if not isinstance(op2, Identity):
      h2 = drop_path(h2, drop_prob)
    s = h1 + h2
    states += [s]

  return tf.concat([states[i] for i in concat], axis=-1) # TODO clean this





class AuxiliaryHeadCIFAR(DataFormatHandler):
  def __init__(self, layers,  num_classes):
    """assuming input size 8x8"""
    super().__init__()
    self.features = Sequential([
        layers.ReLU(),
        layers.AveragePooling2D(5, strides=3, padding='valid'),
        layers.Conv2D(128, 1, kernel_initializer='he_uniform', use_bias=False),
        layers.BatchNormalization(axis=self.axis),
        layers.ReLU(),
        layers.Conv2D(768, 2, kernel_initializer='he_uniform', use_bias=False),
        layers.BatchNormalization(axis=self.axis),
        layers.ReLU()
    ])
    self.classifier = layers.Dense(num_classes)

  def call(self, x):
    x = self.features(x)
    x = self.flatten(x)
    x = self.classifier(x)
    return x


class NetworkCIFAR(DataFormatHandler):
  def __init__(self, layers, label_dim, C=48, n_layers=14, auxiliary=False, genotype="PDARTS", train=False):
    super().__init__()
    self.train = train
    self.label_dim = label_dim
    self.genotype = eval("%s" % genotype)  # TODO improve
    self.n_layers = n_layers
    self._auxiliary = auxiliary

    self.stem0 = lambda layers: Sequential([
        layers.Conv2D(C // 2, kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_uniform', use_bias=False),
        layers.BatchNormalization(axis=self.axis),
        layers.ReLU(),
        layers.Conv2D(C, kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_uniform', use_bias=False),
        layers.BatchNormalization(axis=self.axis)
    ])

    self.stem1 = lambda layers: Sequential([
        layers.ReLU(),
        layers.Conv2D(C, kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_uniform', use_bias=False),
        layers.BatchNormalization(axis=self.axis)
    ])
    self.C_prev_prev, self.C_prev, self.C_curr = C, C, C

    if auxiliary:
      self.auxiliary_head = lambda layers, C_to_auxiliary, label_dim: AuxiliaryHeadImageNet(layers, C_to_auxiliary, label_dim)
    self.classifier = lambda layers: Sequential([
        tf.keras.layers.Flatten(),
        layers.Dense(self.label_dim, kernel_initializer='he_uniform')
    ])


class NetworkImageNet(DataFormatHandler):
  def __init__(self, layers, label_dim, C=48, n_layers=14, auxiliary=False, genotype="PDARTS", train=False):
    super().__init__()
    self.train = train
    self.label_dim = label_dim
    self.genotype = eval("%s" % genotype)  # TODO improve
    self.n_layers = n_layers
    self._auxiliary = auxiliary

    self.stem0 = lambda layers: Sequential([
        tf.keras.layers.ZeroPadding2D(padding=1),
        layers.Conv2D(C // 2, kernel_size=3, strides=2, padding='VALID', kernel_initializer='he_uniform', use_bias=False),
        layers.BatchNormalization(axis=self.axis),
        layers.ReLU(),
        tf.keras.layers.ZeroPadding2D(padding=1),
        layers.Conv2D(C, kernel_size=3, strides=2, padding='VALID', kernel_initializer='he_uniform', use_bias=False),
        layers.BatchNormalization(axis=self.axis)
    ])

    self.stem1 = lambda layers: Sequential([
        layers.ReLU(),
        tf.keras.layers.ZeroPadding2D(padding=1),
        layers.Conv2D(C, kernel_size=3, strides=2, padding='VALID', kernel_initializer='he_uniform', use_bias=False),
        layers.BatchNormalization(axis=self.axis)
    ])
    self.C_prev_prev, self.C_prev, self.C_curr = C, C, C

    if auxiliary:
      self.auxiliary_head = lambda layers, C_to_auxiliary, label_dim: AuxiliaryHeadImageNet(layers, C_to_auxiliary, label_dim)
    self.classifier = lambda layers: Sequential([
        tf.keras.layers.Flatten(),
        layers.Dense(self.label_dim, kernel_initializer='he_uniform')
    ])


def callback_epoch(epoch, num_epochs, initial_drop_path_prob):
  global drop_path_prob
  tf.keras.backend.set_value(drop_path_prob, initial_drop_path_prob * epoch / num_epochs)


class PdartsImageNet(GenericModel):
  def __init__(self, *args, **kwargs):
    global drop_path_prob
    drop_path_prob = tf.keras.backend.variable(kwargs['args']['drop_path_prob'], name='drop_path_prob')
    # drop_path_prob = tf.keras.backend.variable(args['drop_path_prob'], name='drop_path_prob') # TODO implement this alternative
    super().__init__(*args, **kwargs)
    self.c = 48 # number of channels at the beginning of the network
    self.genotype=eval("PDARTS")
    self.n_layers = 14
    self._auxiliary = False

  def model(self):
    self.x = self.layers().Identity()(self.x) # This op is needed so Upstride can insert its custom op
    input = self.x
    
    # Stem 0
    layers = self.layers()
    x = layers.Conv2D(self.c // 2, kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_uniform', use_bias=False)(input)
    x = layers.BatchNormalization(axis=self.axis)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(self.c, kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_uniform', use_bias=False)(x)
    s0 = layers.BatchNormalization(axis=self.axis)(x)

    # Stem 1
    x = layers.ReLU()(s0)
    x = layers.Conv2D(self.c, kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_uniform', use_bias=False)(x)
    s1 = layers.BatchNormalization(axis=self.axis)(x)

    C_curr = self.c


    reduction_prev = True
    for i in range(self.n_layers):
      if i in [self.n_layers // 3, 2 * self.n_layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(self.layers(), self.genotype, C_curr, reduction, reduction_prev)
      s0, s1 = s1, cell(s0, s1, drop_path_prob)

      reduction_prev = reduction
      if i == 2 * self.n_layers // 3 and self._auxiliary and self.train:
        self.logits_aux = self.auxiliary_head(self.layers(), s1)

    self.x = self.layers().AveragePooling2D(7)(s1)
    self.x = layers.Flatten()(self.x)
    # return self.x, logits_aux # TODO handle logits_aux


  def auxiliary_head(self, layers, input_tensor):
    """assuming input size 14x14"""
    x = layers.ReLU()(input_tensor)
    x = layers.AveragePooling2D(5, strides=2, padding='valid')(x)
    x = layers.Conv2D(128, 1, kernel_initializer='he_uniform', use_bias=False)(x)
    x = layers.BatchNormalization(axis=self.axis)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(768, 2, kernel_initializer='he_uniform', use_bias=False)(x)
    x = layers.BatchNormalization(axis=self.axis)(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(self.num_classes)(x)
    return x



class PdartsCIFAR(GenericModel):
  def __init__(self, args, **kwargs):
    global drop_path_prob
    # drop_path_prob = tf.keras.backend.variable(kwargs['args']['drop_path_prob'], name='drop_path_prob')
    drop_path_prob = tf.keras.backend.variable(args['drop_path_prob'], name='drop_path_prob') # TODO implement this alternative
    self.c = 36 # number of channels at the beginning of the network
    self.genotype=eval("PDARTS")
    self.n_layers = 20
    self._auxiliary = False
    super().__init__(args, **kwargs)

  def model(self):
    # Stem
    self.axis = -1 # TODO correct for channel first

    layers = self.layers()
    x = layers.Conv2D(self.c * 3, kernel_size=3, padding='SAME', kernel_initializer='he_uniform', use_bias=False)(self.x)
    s0 = layers.BatchNormalization(axis=self.axis)(x)
    s1 = s0

    C_curr = self.c


    reduction_prev = False
    for i in range(self.n_layers):
      if i in [self.n_layers // 3, 2 * self.n_layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      s0, s1 = s1, cell(self.layers(), self.genotype, C_curr, reduction, reduction_prev, s0, s1, drop_path_prob)

      reduction_prev = reduction
      if i == 2 * self.n_layers // 3 and self._auxiliary and self.train:
        self.logits_aux = self.auxiliary_head(self.layers(), s1)

    self.x = self.layers().AveragePooling2D(1)(s1)
    self.x = layers.Flatten()(self.x)
    # return self.x, logits_aux # TODO handle logits_aux


  def auxiliary_head(self, layers, input_tensor):
    """assuming input size 14x14"""
    x = layers.ReLU()(input_tensor)
    x = layers.AveragePooling2D(5, strides=3, padding='valid')(x)
    x = layers.Conv2D(128, 1, kernel_initializer='he_uniform', use_bias=False)(x)
    x = layers.BatchNormalization(axis=self.axis)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(768, 2, kernel_initializer='he_uniform', use_bias=False)(x)
    x = layers.BatchNormalization(axis=self.axis)(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(self.label_dim)(x)
    return x
