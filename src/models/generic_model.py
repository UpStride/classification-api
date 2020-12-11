import tensorflow as tf

framework_list = [
    "tensorflow",
    "upstride_type0",
    "upstride_type1",
    "upstride_type2",
    "upstride_type3",
    "mix_type0",
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

  def __call__(self,):
    if self.framework == "tensorflow":
      return self.tf_layers
    elif 'upstride' in self.framework:
      return self.up_layers
    # At this point, self.framework is "mix_*", so we need to count the number of layers in the network
    # to know when to shift engine
    self.n_layers += 1
    if self.n_layers - 1 < self.n_layers_before_tf:
      return self.up_layers
    return self.tf_layers


class GenericModel:
  def __init__(self, args, load_searched_arch=None):
    """[summary]

    Args:
        args: Dict containing
          - framework
          - factor : Defaults to 1.
          - input_size : Defaults to (224, 224, 3).
          - n_layers_before_tf
          - name
          - drop_path_prob
          - num_classes (int): Defaults to 1000.
          - conversion_params (dict): containing the params for TF2UpStride & UpStride2TF conversion
        cpu (bool, optional): [description]. Defaults to False.
        hp ([type], optional): [description]. Defaults to None.
        load_searched_arch (str, optional): Yaml file that provide the model definition found by DNAS. Defaults to None.
    """
    self.hp = None  # for keras tuner
    self._layers = Layer(args['framework'], args['n_layers_before_tf'])
    self._previous_layer = tf.keras.layers
    inputs = tf.keras.layers.Input(shape=args['input_size'])
    self.x = inputs
    conversion_params = args['conversion_params']
    self.output_layer_before_up2tf = conversion_params['output_layer_before_up2tf']
    self.tf2up_strategy = conversion_params['tf2up_strategy']
    self.up2tf_strategy = conversion_params['up2tf_strategy']
    self.weight_regularizer = None

    # if the model use auxiliary logits then it will set this variable to a different value than None
    self.logits_aux = None
    # if the model use other inputs then it will set this variable to the list of input to add
    self.inputs = []
    # if the model use custom keras Model then overide this
    self.model_class = tf.keras.Model

    self.factor = args['factor']
    self.label_dim = args['num_classes']
    self.model()

    # output is the list of the vectors to use to compute classification losses
    output_tensors = [self.x]
    if self.logits_aux is not None:
      output_tensors.append(self.logits_aux)

    output_logits = []
    for i, output_tensor in enumerate(output_tensors):
      x = output_tensor
      if self.output_layer_before_up2tf:
        x = self.layers().Dense(self.label_dim, use_bias=True, name=f'Logits_{i}', kernel_regularizer=self.weight_regularizer)(x)
      # Upstride to TF
      if self._previous_layer != tf.keras.layers:
        x = self._previous_layer.Upstride2TF(self.up2tf_strategy)(x)
      if not self.output_layer_before_up2tf:
        x = tf.keras.layers.Dense(self.label_dim, use_bias=True, name=f'Logits_{i}', kernel_regularizer=self.weight_regularizer)(x)
      x = tf.keras.layers.Activation("softmax", dtype=tf.float32)(x)  # dtype float32 is important because of mixed precision
      output_logits.append(x)
    self.model = self.model_class([inputs] + self.inputs, output_logits)
    # self.model = tf.keras.Model(inputs, output_logits)

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
