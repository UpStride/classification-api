import sys
import unittest
from src.data.test_augmentations import TestAugmentations
from src.data.test_dataloader import TestDataLoader
from src.test_losses import TestLosses
from src.models.test_fbnetv2 import *
from src.models.test_fbnet_mobilenet import * 

# from src.test_utils import TestUtils
# from src.models.test_generic_model import TestModel1 # TestLayer
# from src.test_export import TestExport
# from src.test_model_tools import TestLRDecay
# from src.test_metrics import TestMetrics, TestCountFlops

sys.path.append('scripts')

from scripts.test_tfrecord_writer import TestTfrecordWriter


if __name__ == "__main__":
  unittest.main()
