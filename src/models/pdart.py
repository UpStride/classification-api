import tensorflow as tf
from .generic_model import GenericModel
import tensorflow.keras as tfk
from tensorflow.keras import activations, Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, AveragePooling2D, Dense, ReLU, MaxPool2D, DepthwiseConv2D
from collections import namedtuple

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
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: AveragePooling2D(3, strides=stride, padding='SAME'),
    'max_pool_3x3': lambda C, stride, affine: MaxPool2D(3, strides=stride, padding='SAME'),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: Conv_7x1_1x7(C, stride, affine),
}

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1. - drop_prob
    mask = tf.keras.backend.random_bernoulli([x.shape[0], 1, 1, 1], p=keep_prob)
    x /= keep_prob
    x *= mask
  return x
 

class DataFormatHandler(Layer):
  def __init__(self, data_format='channels_first'): # TODO improve data_format handling
    super(DataFormatHandler, self).__init__()
    self.data_format = data_format
    self.is_channels_first = True if self.data_format == 'channels_first' else False
    self.axis = 1 if self.is_channels_first else -1


class Identity(DataFormatHandler):
  def __init__(self):
    super(Identity, self).__init__()

  def call(self, x):
    return x


class Zero(DataFormatHandler): # TODO
  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def call(self, x):
    if self.is_channels_first:
      n, c, h, w = x.shape
    else:
      n, h, w, c = x.shape
    h //= self.stride
    w //= self.stride
    padding = tf.zeros([n, c, h, w]) if self.is_channels_first else tf.zeros([n, h, w, c])
    return padding


class SepConv(DataFormatHandler): # TODO
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    kwargs = {'data_format':self.data_format}
    self.op = Sequential([
      ReLU(),
      DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, depthwise_initializer='he_uniform', use_bias=False, **kwargs),
      Conv2D(C_in, kernel_size=1, padding='VALID', kernel_initializer='he_uniform', use_bias=False, **kwargs),
      BatchNormalization(axis=self.axis, trainable=affine),
      ReLU(),
      DepthwiseConv2D(kernel_size=kernel_size, strides=1, padding=padding, depthwise_initializer='he_uniform', use_bias=False, **kwargs),
      Conv2D(C_out, kernel_size=1, padding='VALID', kernel_initializer='he_uniform', use_bias=False, **kwargs),
      BatchNormalization(axis=self.axis, trainable=affine)
    ])

  def call(self, x):
    return self.op(x)


class Conv_7x1_1x7(DataFormatHandler): # TODO
  def __init__(self, C, stride, affine):
    super(Conv_7x1_1x7, self).__init__()
    kwargs = {'data_format':self.data_format}
    self.op = Sequential([
      ReLU(),
      Conv2D(C, (1, 7), strides=(1, stride), padding=(0, 3), kernel_initializer='he_uniform', use_bias=False, **kwargs), # TODO fix padding
      Conv2D(C, (7, 1), strides=(stride, 1), padding=(3, 0), kernel_initializer='he_uniform', use_bias=False, **kwargs), # TODO fix padding
      BatchNormalization(axis=self.axis, trainable=affine)
    ])

  def call(self, x):
    return self.op(x)


class DilConv(DataFormatHandler):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    kwargs = {'data_format':self.data_format}
    self.op = Sequential([
      ReLU(),
      DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation, depthwise_initializer='he_uniform', use_bias=False, **kwargs),
      Conv2D(C_out, kernel_size=1, padding='VALID', kernel_initializer='he_uniform', use_bias=False, **kwargs),
      BatchNormalization(axis=self.axis, trainable=affine)
    ])

  def call(self, x):
    return self.op(x)


class FactorizedReduce(DataFormatHandler):
  def __init__(self, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = ReLU()
    self.conv_1 = Conv2D(C_out // 2, 1, strides=2, padding='valid', data_format=self.data_format, kernel_initializer='he_uniform', use_bias=False)
    self.conv_2 = Conv2D(C_out // 2, 1, strides=2, padding='valid', data_format=self.data_format, kernel_initializer='he_uniform', use_bias=False)
    self.bn = BatchNormalization(axis_bn=self.axis, trainable=affine)

  def call(self, x):
    x = self.relu(x)
    to_be_concat = [self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])] if self.is_channels_first else [self.conv_1(x), self.conv_2(x[:, 1:, 1:, :])]
    out = tf.concat(values=to_be_concat, axis=self.axis)
    out = self.bn(out)
    return out


class ReLUConvBN(DataFormatHandler):
  def __init__(self, C_out, kernel_size, stride, padding, affine=True): # TODO handle explicit padding vs ['same', 'valid']
    super(ReLUConvBN, self).__init__()
    self.op = Sequential([
      ReLU(),
      Conv2D(C_out, kernel_size, strides=stride, padding=padding, data_format=self.data_format, kernel_initializer='he_uniform', use_bias=False),
      BatchNormalization(axis=self.axis, trainable=affine)
    ])

  def call(self, x):
    return self.op(x)


class Cell(DataFormatHandler): # TODO inheritance
  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C)
    else:
      self.preprocess0 = ReLUConvBN(C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C, 1, 1, 0)    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = []
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def call(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return tf.concat([states[i] for i in self._concat], axis=self.axis)


class AuxiliaryHeadImageNet(DataFormatHandler):
  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = Sequential([
      ReLU(),
      AveragePooling2D(5, strides=2, padding='valid', data_format=self.data_format),
      Conv2D(128, 1, data_format=self.data_format, kernel_initializer='he_uniform', use_bias=False),
      BatchNormalization(axis=self.axis),
      ReLU(),
      Conv2D(768, 2, data_format=self.data_format, kernel_initializer='he_uniform', use_bias=False),
      BatchNormalization(axis=self.axis),
      ReLU()
    ])
    self.classifier = Dense(num_classes)

  def call(self, x):
    x = self.features(x)
    x = self.classifier(tf.reshape(x, [x.shape[0], -1]))
    return x


class NetworkImageNet(DataFormatHandler):
  def model(self, C=46, num_classes=1000, layers=14, auxiliary=False, genotype="PDARTS"):
  # def __init__(self, C, num_classes, layers, auxiliary, genotype):
    genotype = eval("%s" % genotype)
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    kwargs = {'data_format' : self.data_format}
    self.stem0 = Sequential([
      # tfk.Input(shape=(3, some_height, some_width)), # TBD
      Conv2D(C // 2, kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_uniform', use_bias=False, **kwargs),
      BatchNormalization(axis=self.axis),
      ReLU(),
      Conv2D(C, kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_uniform', use_bias=False, **kwargs),
      BatchNormalization(axis=self.axis)
    ])

    self.stem1 = Sequential([
      ReLU(),
      Conv2D(C, kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_uniform', use_bias=False, **kwargs),
      BatchNormalization(axis=self.axis)
    ])
    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = []
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = AveragePooling2D(7, data_format=self.data_format)
    self.classifier = Dense(num_classes, kernel_initializer='he_uniform')

  def call(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux
