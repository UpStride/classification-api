from typing import List
import tensorflow as tf

"""
Question:
- Weight decay ?
- kwargs for specific stuff ?
"""


def load_upstride(upstride_type: int):
  """This function load one of upstride types 
  """
  if upstride_type == -1:
    return None
  if upstride_type == 0:
    import upstride.type0.tf.keras.layers as up_layers
    return up_layers
  if upstride_type == 1:
    import upstride.type1.tf.keras.layers as up_layers
    return up_layers
  if upstride_type == 2:
    import upstride.type2.tf.keras.layers as up_layers
    return up_layers
  if upstride_type == 3:
    import upstride.type3.tf.keras.layers as up_layers
    return up_layers


class GenericModelBuilder:
  def __init__(self, input_size, changing_ids: List[str], num_classes, factor=1, upstride_type=-1, tf2upstride_strategy="", upstride2tf_strategy="", weight_decay=0, **kwargs):
    self.input_size = input_size
    self.num_classes = num_classes
    self.factor = factor
    self.upstride_type = upstride_type
    self.tf2upstride_strategy = tf2upstride_strategy
    self.upstride2tf_strategy = upstride2tf_strategy

    # Configure list of ids to change framework
    if upstride_type == -1:
      # then no switch between tf and upstride
      self.changing_ids = []
    elif changing_ids == []:
      # then set default parameters
      self.changing_ids = ['beginning', 'end_after_dense']
    else:
      self.changing_ids = changing_ids

    # kwargs contains special parameter that can be specific for one model. For instance
    # - load_searched_arch for architecture search method
    # - drop_path_prob for fb-net
    # - conversion_params if tf2upstride or upstride2tf need specific parameters
    # - hp : the keras-tuner hyperparameters
    self.kwargs = kwargs 

    # self.layers is the layers package to use when building the neural network
    self.layers = tf.keras.layers 
    self.upstride_layers = load_upstride(upstride_type)
    self._is_using_tf_layers = True

    # weight_regularizer can be call in the model definition in any subclass of GenericModel
    self.weight_regularizer = tf.keras.regularizers.l2(l=weight_decay)

    # if the model use custom keras Model then overide this
    # This is usefull for SAM method
    self.model_class = tf.keras.Model

    # if the model use other inputs than the image then it need to add these tensors in this list
    # This is usefull for P-Darts, FB-NET and SAM methods
    self.inputs = []


  def maybe_change_framework(self, id, inputs):
    """ When defining a custom model, this function should be called every time it can make sense to switch
    between tensorflow and upstride

    Args:
      x: can be a tensor or a list of tensors.

    Return: a tensor if x is a tensor, a list of tensors if x is a list of tensors
    """

    inputs_is_single_tensor = False
    if type(inputs) is not list:
      inputs_is_single_tensor = True
      inputs = [inputs]

    if id in self.changing_ids:
      if self._is_using_tf_layers:
        # Then converting from Tensorflow to Upstride
        self._is_using_tf_layers = False
        self.layers = self.upstride_layers

        out_tensors = []
        for x in inputs:
          out_tensors.append(self.upstride_layers.TF2Upstride(self.tf2upstride_strategy)(x))
      else:
        # Then converting from Upstride to Tensorflow
        self._is_using_tf_layers = True
        self.layers = tf.keras.layers

        out_tensors = []
        for x in inputs:
          out_tensors.append(self.upstride_layers.Upstride2TF(self.upstride2tf_strategy)(x))
    else:
      # Don't change the input
      out_tensors = inputs

    if inputs_is_single_tensor:
      out_tensors = out_tensors[0]

    return out_tensors

  def model(self, x):
    raise NotImplementedError("you need to overide method model")

  def build(self):
    inputs = tf.keras.layers.Input(shape=self.input_size)
    x = self.maybe_change_framework("beginning", inputs)
    # output_tensors is the list of the vectors to use to compute classification losses (main output + auxilary losses)
    output_tensors = self.model(x)
    if type(output_tensors) != list:
      output_tensors = [output_tensors]

    output_tensors = self.maybe_change_framework("end_before_dense", output_tensors)

    for i, x in enumerate(output_tensors):
      output_tensors[i] = self.layers.Dense(self.num_classes, use_bias=True, name=f'Logits_{i}', kernel_regularizer=self.weight_regularizer)(x)
    output_tensors = self.maybe_change_framework("end_after_dense", output_tensors)

    for i, x in enumerate(output_tensors):
      output_tensors[i] = tf.keras.layers.Activation("softmax", dtype=tf.float32)(x)  # dtype float32 is important because of mixed precision

    model = self.model_class([inputs], output_tensors)

    return model
