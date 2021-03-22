import unittest

import tensorflow as tf
from src.models import model_name_to_class
from tqdm import tqdm

class TestCompareChannelsFirstLast(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model_kwargs = {
      'input_size': [224, 224, 3],
      'changing_ids': [],
      'num_classes': 10,
    }

    cls.list_of_models = model_name_to_class.values()

    # below model do not work for upstride types
    remove_models = [
      # These models do not match EfficientDet definition. There is a open Pull request yet to be merged. 
      # This should be included once the pull request is merged.
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
    [tmp_list_models.pop(model) for model in remove_models]
    cls.list_models_upstride = tmp_list_models.values()

  def test_compare_model_params_tensorflow(self):
    self.model_kwargs.update({"upstride_type": -1, "factor": 1})
    print("Building models for Channels_first and Channels_last for Tensorflow and Compare") 
    for model in tqdm(self.list_of_models):
      # switch to channels first
      tf.keras.backend.set_image_data_format('channels_first')
      model_NCHW = model(**self.model_kwargs).build()
      model_NCHW_params = model_NCHW.count_params()
      del model_NCHW
      tf.keras.backend.clear_session()
      # switch back to channels last
      tf.keras.backend.set_image_data_format('channels_last')
      model_NHWC = model(**self.model_kwargs).build()
      model_NHWC_params = model_NHWC.count_params()
      del model_NHWC
      tf.keras.backend.clear_session()
      # compare
      # print(f"Model Name    : {model.__name__} TensorFlow")
      # print(f"Channels_last : {model_NHWC_params:,}")
      # print(f"Channels_first: {model_NCHW_params:,}") 
      self.assertEqual(model_NHWC_params, model_NCHW_params)

  def test_compare_model_params_upstride(self):
    for up_type in [1, 2]: 
      self.model_kwargs.update({"upstride_type": up_type, "factor": 2**up_type}) 
      print(f"Building models for Channels_first and Channels_last for Upstride type{up_type} and Compare") 
      for model in tqdm(self.list_models_upstride):
        # switch to channels first
        tf.keras.backend.set_image_data_format('channels_first')
        model_NCHW = model(**self.model_kwargs).build()
        model_NCHW_params = model_NCHW.count_params()
        del model_NCHW
        tf.keras.backend.clear_session()
        # switch back to channels last
        tf.keras.backend.set_image_data_format('channels_last')
        model_NHWC = model(**self.model_kwargs).build()
        model_NHWC_params = model_NHWC.count_params()
        del model_NHWC
        tf.keras.backend.clear_session()
        # compare
        # print(f"Model Name    : {model.__name__} UpStride type{up_type}")
        # print(f"Channels_last : {model_NHWC_params:,}")
        # print(f"Channels_first: {model_NCHW_params:,}") 
        self.assertEqual(model_NHWC_params, model_NCHW_params)