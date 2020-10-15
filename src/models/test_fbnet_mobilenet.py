import unittest
import yaml 
import tempfile
import numpy as np

from .fbnet_mobilenet import FBNet_MobileNetV2Imagenet
import tensorflow as tf 

class TestFBnetMobileNet(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.img = np.ones((1, 224, 224, 3), dtype=np.float32)
  
    cls.test_mapping = {
      "conv2d_01": 24, 
      "irb_01": 20, 
      "irb_02": 40, 
      "irb_03": 32, 
      "irb_04": 40, 
      "irb_05": 80, 
      "irb_06": 64, 
      "irb_07": 80, 
      "irb_08": 160, 
      "irb_09": 96, 
      "irb_10": 152, 
      "irb_11": 224, 
      "irb_12": 136, 
      "irb_13": 224, 
      "irb_14": 160, 
      "irb_15": 352, 
      "irb_16": 368, 
      "irb_17": 336
    }

    cls.tempdir = tempfile.mkdtemp()
    cls.file_path = cls.tempdir + '/test.yaml'
    with open(cls.file_path,'w') as f:
      yaml.dump(cls.test_mapping,f)
    
    cls.channel_last = True # TODO test for channels first
  
  def test_init(self):
    print(self.img[1:])
    model = FBNet_MobileNetV2Imagenet(
      'tensorflow',
      factor=1,
      input_shape=self.img.shape[1:],
      label_dim=10,
      load_searched_arch=self.file_path).model
    
    model.summary()
    get_dict = {}
    for layer in model.layers:
      # This type of checking the channels based on the architecture is not ideal.
      # For this specific case we use the projection of the MobileNet block to get the channels used. 
      if layer.name.startswith('conv2d_01') or layer.name.endswith('project'): # not proud
        get_dict[layer.name.split('project')[0]] = layer.output.shape[-1] if self.channel_last else layer.output.shape[1]
    
    self.assertDictEqual(get_dict,self.test_mapping)