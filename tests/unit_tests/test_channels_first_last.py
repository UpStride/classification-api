import unittest

import tensorflow as tf
from src.models import model_name_to_class

class TestCompareChannelsFirstLast(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.kwargs = {
      'input_size': [224, 224, 3],
      'changing_ids': [],
      'num_classes': 10,
    }

    cls.list_of_models = model_name_to_class.values()

    # below model do not work for upstride types
    remove_models = [
      # The channel_last and first values mismatch only for upstride types. 
      # FIXME when this is prioritized.
      "EfficientNetB0", 
      "EfficientNetB1",
      "EfficientNetB2",
      "EfficientNetB3",
      "EfficientNetB4",
      "EfficientNetB5",
      "EfficientNetB6",
      "EfficientNetB7",
      # SeparableConv2D is not supported for upstride types
      "NASNetCIFAR",
      "NASNetLarge",
      "NASNetMobile",
    ]

    tmp_list_models = model_name_to_class
    # remove models that are not supported
    [tmp_list_models.pop(i) for i in remove_models]
    cls.list_models_upstride = tmp_list_models.values()

  def test_compare_model_params_tensorflow(self):
    self.kwargs.update({"upstride_type": -1, "factor": 1}) 
    for model1, model2 in zip(self.list_of_models, self.list_of_models):
      if model1 == model2: # would be same, just a safety check
        # switch to channels first
        tf.keras.backend.set_image_data_format('channels_first')
        model_NCHW = model1(**self.kwargs).build()
        model_NCHW_params = model_NCHW.count_params()
        # switch back to channels last
        tf.keras.backend.set_image_data_format('channels_last')
        model_NHWC = model2(**self.kwargs).build()
        model_NHWC_params = model_NHWC.count_params()
        # compare
        print(f"Model Name    : {model1.__name__} TensorFlow")
        print(f"Channels_last : {model_NHWC_params:,}")
        print(f"Channels_first: {model_NCHW_params:,}") 
        self.assertEqual(model_NHWC_params, model_NCHW_params)

  def test_compare_model_params_upstride(self):
    for up_type in [2]: #TODO add type1 later. resnet model parameters do not match. FIXME when this is prioritized.
      self.kwargs.update({"upstride_type": up_type, "factor": 2**up_type}) 
      for model1, model2 in zip(self.list_models_upstride, self.list_models_upstride):
        if model1 == model2: # would be same, just a safety check
          # switch to channels first
          tf.keras.backend.set_image_data_format('channels_first')
          model_NCHW = model1(**self.kwargs).build()
          model_NCHW_params = model_NCHW.count_params()
          # switch back to channels last
          tf.keras.backend.set_image_data_format('channels_last')
          model_NHWC = model2(**self.kwargs).build()
          model_NHWC_params = model_NHWC.count_params()
          # compare
          print(f"Model Name    : {model1.__name__} UpStride type{up_type}")
          print(f"Channels_last : {model_NHWC_params:,}")
          print(f"Channels_first: {model_NCHW_params:,}") 
          self.assertEqual(model_NHWC_params, model_NCHW_params)