import tensorflow as tf

framework_list = [
    "tensorflow",
    "upstride_type0",
    "upstride_type1",
    "upstride_type2",
    "upstride_type3",
    "mix_real",
    "mix_type1",
    "mix_type2",
    "mix_type3",
]


class Layer:
  def __init__(self, framework, n_layers_before_tf):
    self.tf_layers = tf.keras.layers
    self.up_layers = None
    self.n_layers = 0
    self.n_layers_before_tf = n_layers_before_tf
    self.framework = framework

    if framework != "tensorflow":
      if "0" in framework:
        import upstride.type0.tf.keras.layers as up_layers
        self.up_layers = up_layers
      if "1" in framework:
        import upstride.type1.tf.keras.layers as up_layers
        self.up_layers = up_layers
      if "2" in framework:
        import upstride.type2.tf.keras.layers as up_layers
        self.up_layers = up_layers
      if "3" in framework:
        import upstride.type3.tf.keras.layers as up_layers
        self.up_layers = up_layers
      # patches to enable layers not in the cpp engine
      # register the attribute manually at runtime
      layers_to_register = ['Dropout']
      for l in layers_to_register:
        try:
          a = getattr(up_layers, l)  # same as up_layers.Dropout, for instance
        except AttributeError as e:
          setattr(up_layers, l, getattr(tf.keras.layers, l))

  def __call__(self,):
    if self.framework == "tensorflow":
      return self.tf_layers
    elif 'upstride' in self.framework:
      return self.up_layers
    self.n_layers += 1
    if self.n_layers - 1 < self.n_layers_before_tf:
      return self.up_layers
    return self.tf_layers

  def __check_framework(self, framework):
    if framework not in framework_list:
      raise ValueError("Unknown framework type: {}".format(framework))


class GenericModel:
  def __init__(self, framework: str, conversion_params, factor=1, input_shape=(224, 224, 3), label_dim=1000, n_layers_before_tf=0, cpu=False, hp=None, load_searched_arch=None, args=None):
    """[summary]

    Args:
        conversion_params (dict): containing the params for TF2UpStride & UpStride2TF conversion
        framework (str): [description]
        factor (int, optional): [description]. Defaults to 1.
        input_shape (tuple, optional): [description]. Defaults to (224, 224, 3).
        label_dim (int, optional): [description]. Defaults to 1000.
        n_layers_before_tf (int, optional): [description]. Defaults to 0.
        cpu (bool, optional): [description]. Defaults to False.
        hp ([type], optional): [description]. Defaults to None.
        load_searched_arch (str, optional): Yaml file that provide the model definition found by DNAS. Defaults to None.
    """
    self.hp = hp  # for keras tuner
    self._layers = Layer(framework, n_layers_before_tf)
    self._previous_layer = tf.keras.layers
    inputs = tf.keras.layers.Input(shape=input_shape)
    self.x = inputs
    self.output_layer_before_up2tf = conversion_params['output_layer_before_up2tf']
    self.tf2up_strategy = conversion_params['tf2up_strategy']
    self.up2tf_strategy = conversion_params['up2tf_strategy']
    self.weight_regularizer = None

    self.factor = factor
    self.label_dim = label_dim
    if cpu:
      with tf.device('/CPU:0'):
        self.model()
    else:
      self.model()

    if self.output_layer_before_up2tf:
      self.x = self.layers().Dense(self.label_dim, use_bias=True, name='logits_before_up2tf', kernel_regularizer=self.weight_regularizer)(self.x)

    # Upstride to TF
    if self._previous_layer != tf.keras.layers:
      self.x = self._previous_layer.Upstride2TF(self.up2tf_strategy)(self.x)

    if not self.output_layer_before_up2tf:
      self.x = tf.keras.layers.Dense(self.label_dim, use_bias=True, name='logits_after_up2tf', kernel_regularizer=self.weight_regularizer)(self.x)

    x = tf.keras.layers.Activation("softmax", dtype=tf.float32)(self.x)  # dtype float32 is important because of mixed precision
    self.model = tf.keras.Model(inputs, x)

  def layers(self):
    """return the layer to use and automatically convert between tensorflow and upstride
    """
    l = self._layers()
    if l != self._previous_layer and self._previous_layer == tf.keras.layers:
      # then switch from tf to upstride
      self.x = l.TF2Upstride(self.tf2up_strategy)(self.x)
    if l != self._previous_layer and l == tf.keras.layers:
      # then switch from upstride to tf
      self.x = self._previous_layer.Upstride2TF(self.up2tf_strategy)(self.x)
    self._previous_layer = l
    return l

  def model(self):
    raise NotImplementedError("you need to overide method model")
