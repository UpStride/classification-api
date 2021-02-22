import tensorflow as tf
import numpy as np 
from .generic_model import GenericModelBuilder
from tensorflow.keras import Sequential
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
    'none': lambda layers, C, strides, layers_conf, trainable: Zero(strides),
    'avg_pool_3x3': lambda layers, C, strides, layers_conf, trainable: AvgPool(layers, strides),
    'max_pool_3x3': lambda layers, C, strides, layers_conf, trainable: MaxPool(layers, strides),
    'skip_connect': lambda layers, C, strides, layers_conf, trainable: Identity() if strides == 1 else FactorizedReduce(layers, C, trainable=trainable),
    'sep_conv_3x3': lambda layers, C, strides, layers_conf, trainable: SepConv(layers, C, C, 3, strides, 1, layers_conf, trainable=trainable),
    'sep_conv_5x5': lambda layers, C, strides, layers_conf, trainable: SepConv(layers, C, C, 5, strides, 2, layers_conf, trainable=trainable),
    'sep_conv_7x7': lambda layers, C, strides, layers_conf, trainable: SepConv(layers, C, C, 7, strides, 3, layers_conf, trainable=trainable),
    'dil_conv_3x3': lambda layers, C, strides, layers_conf, trainable: DilConv(layers, C, 3, strides, 2, 2, layers_conf, trainable=trainable),
    'dil_conv_5x5': lambda layers, C, strides, layers_conf, trainable: DilConv(layers, C, 5, strides, 4, 2, layers_conf, trainable=trainable),
    'conv_7x1_1x7': lambda layers, C, strides, layers_conf, trainable: Conv_7x1_1x7(layers, C, strides, layers_conf, trainable),
}


class DropPath(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, inputs):
    x, drop_prob = inputs

    keep_prob = 1. - drop_prob
    shape = [tf.shape(x)[0], 1, 1, 1]
    # shape = [1, 1, 1, 1]  # this version is simpler to debug
    mask_dist = tf.random.uniform(shape, dtype=x.dtype)
    mask = tf.math.floor(mask_dist + keep_prob)  # This line emulates tf.keras.backend.random_bernoulli
    x = tf.math.divide(x, keep_prob)
    x = tf.math.multiply(x, mask) 
    return x


class DataFormatHandler(tf.keras.Model):
  """ This class is needed because some Keras layer don't look at tf.keras.backend.image_data_format
  to know if there are channels first or last (for instance BatchNorm....)
  """

  def __init__(self):
    super().__init__()
    self.is_channels_first = True if tf.keras.backend.image_data_format() == 'channels_first' else False
    self.axis = 1 if self.is_channels_first else -1


# Primitives functions of P-Darts
class Identity(DataFormatHandler):
  def __init__(self):
    super().__init__()

  def call(self, input_tensor, training=False):
    return input_tensor


class Zero(DataFormatHandler):
  def __init__(self, strides):
    super().__init__()
    self.strides = strides

  def call(self, input_tensor, training=False):
    if self.is_channels_first:
      n, c, h, w = input_tensor.shape
    else:
      n, h, w, c = input_tensor.shape
    h //= self.strides
    w //= self.strides
    output = tf.zeros([n, c, h, w]) if self.is_channels_first else tf.zeros([n, h, w, c])
    return output


class AvgPool(DataFormatHandler):
  def __init__(self, layers, strides):
    super().__init__()
    self.avgpool = layers.AveragePooling2D(3, strides=strides, padding='SAME')

  def call(self, input_tensor, training=False):
    x = self.avgpool(input_tensor)
    return x


class MaxPool(DataFormatHandler):
  def __init__(self, layers, strides):
    super().__init__()
    self.maxpool = layers.MaxPool2D(3, strides=strides, padding='SAME')

  def call(self, input_tensor, training=False):
    x = self.maxpool(input_tensor)
    return x


class SepConv(DataFormatHandler):
  def __init__(self, layers, C_in, C_out, kernel_size, strides, padding, layers_conf, trainable=True):
    super().__init__()
    self.C_in = C_in
    self.C_out = C_out
    self.kernel_size = kernel_size
    self.stride = strides
    self.padding = padding
    self.affine = trainable

    self.sub_model = tf.keras.Sequential([
        layers.ReLU(),
        layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='SAME', **layers_conf['depthwise']),
        layers.Conv2D(C_in, kernel_size=1, padding='SAME', **layers_conf['conv']),
        layers.BatchNormalization(trainable=trainable, **layers_conf['bn']),
        layers.ReLU(),
        layers.DepthwiseConv2D(kernel_size=kernel_size, strides=1, padding='SAME', **layers_conf['depthwise']),
        layers.Conv2D(C_out, kernel_size=1, padding='SAME', **layers_conf['conv']),
        layers.BatchNormalization(trainable=trainable, **layers_conf['bn'])
    ])

  def call(self, input_tensor, training=False):
    return self.sub_model(input_tensor)


class Conv_7x1_1x7(DataFormatHandler):
  def __init__(self, layers, C, strides, layers_conf, trainable):
    super().__init__()
    self.sub_model = tf.keras.Sequential([
        layers.ReLU(),
        layers.Conv2D(C, (1, 7), strides=(1, strides), padding='SAME', **layers_conf['conv']),
        layers.Conv2D(C, (7, 1), strides=(strides, 1), padding='SAME', **layers_conf['conv']),
        layers.BatchNormalization(trainable=trainable, **layers_conf['bn'])
    ])

  def call(self, input_tensor, training=False):
    return self.sub_model(input_tensor)


class DilConv(DataFormatHandler):
  def __init__(self, layers, C_out, kernel_size, strides, padding, dilation, layers_conf, trainable=True):
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
      self.dw_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=1, padding='VALID', dilation_rate=1, **layers_conf['depthwise'])
    else:
      self.dw_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='VALID', dilation_rate=dilation, **layers_conf['depthwise'])
    self.conv = layers.Conv2D(C_out, kernel_size=1, padding='VALID', **layers_conf['conv'])
    self.bn = layers.BatchNormalization(trainable=trainable, **layers_conf['bn'])

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
  def __init__(self, layers, C_out, layers_conf, trainable=True):
    super().__init__()
    assert C_out % 2 == 0
    self.relu = layers.ReLU()
    self.conv_1 = layers.Conv2D(C_out // 2, 1, strides=2, padding='VALID', **layers_conf['conv'])
    self.conv_2 = layers.Conv2D(C_out // 2, 1, strides=2, padding='VALID', **layers_conf['conv'])
    self.bn = layers.BatchNormalization(trainable=trainable, **layers_conf['bn'])

  def call(self, x):
    x = self.relu(x)
    to_be_concat = [self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])] if self.is_channels_first else [self.conv_1(x), self.conv_2(x[:, 1:, 1:, :])]
    out = tf.concat(values=to_be_concat, axis=self.axis)
    out = self.bn(out)
    return out


class ReLUConvBN(DataFormatHandler):
  def __init__(self, layers, C_out, kernel_size, strides, padding, layers_conf, trainable=True):
    """
    PyTorch code:
    self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(C_out, affine=affine)
    )

    The PyTorch code has parameters (kernel_size, strides, padding) always equal to (1, 1, 0)

    We kept the same signature but the padding parameter is useless as we always work with padding='VALID'. 
    If the user call this function with padding != 0 then raise an error

    """
    super().__init__()
    assert padding == 0, "padding need to be 0"
    self.relu = layers.ReLU()
    self.conv2d = layers.Conv2D(C_out, kernel_size, strides=strides, padding='VALID', **layers_conf['conv'])
    self.bn = layers.BatchNormalization(trainable=trainable, **layers_conf['bn'])

  def call(self, x):
    x = self.relu(x)
    x = self.conv2d(x)
    x = self.bn(x)
    return x


class Cell(DataFormatHandler):
  def __init__(self, layers, genotype, C, reduction, reduction_prev, layers_conf):
    """
    Implementation note: 
    - The pytorch code has parameters C_prev_prev and C_prev. Here we don't need them as keras auto-infer
    input shape
    """
    super().__init__()

    # function that preprocess s0 and s1
    self.preprocess0 = FactorizedReduce(layers, C, layers_conf) if reduction_prev else ReLUConvBN(layers, C, 1, 1, 0, layers_conf)
    self.preprocess1 = ReLUConvBN(layers, C, 1, 1, 0, layers_conf)

    # find the good cell definition
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    assert len(op_names) == len(indices)  # Sanity check
    self._indices = indices

    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = []
    for name, index in zip(op_names, indices):
      strides = 2 if reduction and index < 2 else 1
      op = OPS[name](layers, C, strides, layers_conf, True)
      self._ops += [op]

    # define the 2 droppath function in case it's usefull
    # TODO : maybe define these fonction only if needed ?
    self.drop_path_1 = DropPath()
    self.drop_path_2 = DropPath()

  def call(self, input_layers, training=False):
    s0, s1, drop_path_prob = input_layers
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

      if not isinstance(op1, Identity) and training:
        h1 = self.drop_path_1([h1, drop_path_prob])
      if not isinstance(op2, Identity) and training:
        h2 = self.drop_path_2([h2, drop_path_prob])
      s = h1 + h2
      states += [s]

    return tf.concat([states[i] for i in self._concat], axis=self.axis)


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


class PdartsImageNet(GenericModelBuilder):
  def __init__(self, *args, **kwargs):
    define_drop_path_prob(kwargs['args']['drop_path_prob'])
    super().__init__(*args, **kwargs)
    self.c = 48  # number of channels at the beginning of the network
    self.genotype = eval("PDARTS")
    self.n_layers = 14
    self._auxiliary = False

  def model(self, x):
    x = self.layers.Identity()(x)  # This op is needed so Upstride can insert its custom op
    input = x

    # Stem 0
    layers = self.layers
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
      cell = Cell(self.layers, self.genotype, C_curr, reduction, reduction_prev)
      s0, s1 = s1, cell(s0, s1, drop_path_prob)

      reduction_prev = reduction
      if i == 2 * self.n_layers // 3 and self._auxiliary and self.train:
        self.logits_aux = self.auxiliary_head(self.layers, s1)

    x = self.layers.AveragePooling2D(7)(s1)
    x = layers.Flatten()(x)
    return x, logits_aux

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


class PdartsCIFAR(GenericModelBuilder):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # define_drop_path_prob(kwargs['args']['drop_path_prob'])
    self.c = 36  # number of channels at the beginning of the network
    self.genotype = eval("PDARTS")
    self.n_layers = 20
    self._auxiliary = True

    # specify that the compilation will not be done with tf.keras.Model but PDartsModel
    self.model_class = PDartsModel

    # general configuration of some layers
    weight_regularizer = tf.keras.regularizers.l2(l=0.0003)

    self.bn_conf = {
        'axis': -1,
        'momentum': 0.9,
        'epsilon': 1e-5,
    }

    self.conv_conf = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'kernel_regularizer': weight_regularizer
    }

    self.depthwise_conf = {
        'depthwise_initializer': 'he_uniform',
        'use_bias': False,
        'depthwise_regularizer': weight_regularizer
    }

    self.layers_conf = {
        'bn': self.bn_conf,
        'conv': self.conv_conf,
        'depthwise': self.depthwise_conf
    }

  def model(self, x):
    drop_path_prob = tf.keras.layers.Input([])
    self.inputs.append(drop_path_prob)
    # Stem
    self.axis = -1  # TODO correct for channel first

    layers = self.layers
    x = self.layers.Conv2D(self.c * 3, kernel_size=3, padding='SAME', **self.conv_conf)(x)
    s0 = layers.BatchNormalization(**self.bn_conf)(x)
    s1 = s0

    C_curr = self.c

    reduction_prev = False
    for i in range(self.n_layers):
      if i in [self.n_layers // 3, 2 * self.n_layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      s0, s1 = s1, Cell(self.layers, self.genotype, C_curr, reduction, reduction_prev, self.layers_conf)([s0, s1, drop_path_prob])

      reduction_prev = reduction
      if i == 2 * self.n_layers // 3 and self._auxiliary:
        logits_aux = self.auxiliary_head(self.layers, s1)

    x = self.layers.GlobalAveragePooling2D()(s1)
    if self._auxiliary:
      return [x, logits_aux]
    else:
      return x

  def auxiliary_head(self, layers, input_tensor):
    """assuming input size 14x14"""
    x = layers.ReLU()(input_tensor)
    x = layers.AveragePooling2D(5, strides=3, padding='valid')(x)
    x = layers.Conv2D(128, 1, **self.conv_conf)(x)
    x = layers.BatchNormalization(**self.bn_conf)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(768, 2, **self.conv_conf)(x)
    x = layers.BatchNormalization(**self.bn_conf)(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    return x


class PDartsModel(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.drop_path_prob = 0
  
  @tf.function
  def train_step(self, data, drop_path_prob):
    """ see https://keras.io/guides/customizing_what_happens_in_fit/
    """
    x, y = data

    with tf.GradientTape() as tape:
      y_pred = self([x, drop_path_prob], training=True)
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.compiled_metrics.update_state(y, y_pred)
    output = {m.name: m.result() for m in self.metrics}
    output['drop_path_prob'] = drop_path_prob
    return output

  @tf.function
  def test_step(self, data):
    """ see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py#L1189
    """
    x, y = data
    
    y_pred = self([x, 0.], training=False)
    self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}

  def on_epoch_begin_callback(self, current_epoch: int, max_epoch: int, max_drop_path_prob: float):
    """ Callback to update drop_path_prob during the training
    """
    self.drop_path_prob = max_drop_path_prob * current_epoch / max_epoch

  def fit(self, x, validation_data, epochs, callbacks, initial_epoch):
    """
    Notes regarding mixed-precision
    https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/engine/training.py#L551

    in tf 2.4, it seems that user don't need to do anything special in the custom training loop anymore. 
    Indeed, during compilation, if mixed-precision is enable, then the optimizer is wrapped in a lossScaleOptimizer
    that take care of everything

    """
    # TODO have same output than Keras

    callbacks = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=self)

    # call all callbacks at the beginning of the training
    callbacks.on_train_begin()

    with self.distribute_strategy.scope():
      for epoch in range(initial_epoch, epochs+1):
        self.reset_metrics()
        callbacks.on_epoch_begin(epoch)

        # Training
        print('\n\n\nTraining')
        for step, training_point in enumerate(x):
          callbacks.on_train_batch_begin(step)
          # logs = self.train_step(training_point)
          logs = self.distribute_strategy.run(self.train_step, args=(training_point, self.drop_path_prob))

          if self.stop_training:
            break
          print(logs)
          callbacks.on_train_batch_end(step)

        # Validation
        print('Validation')
        for step, validation_point in enumerate(validation_data):
          # logs = self.test_step(validation_point)
          logs = self.distribute_strategy.run(self.test_step, args=(validation_point,))
          print(logs)

        callbacks.on_epoch_end(epoch)

    callbacks.on_train_end()
