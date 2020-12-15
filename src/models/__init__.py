import sys
import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from .alexnet import AlexNet, AlexNetQ, AlexNetToy, AlexNetNCHW
from .mobilenet import MobileNetV2, MobileNetV2NCHW, MobileNetV2Cifar10, MobileNetV2Cifar10_2, MobileNetV2Cifar10Hyper, MobileNetV2Cifar10NCHW
from .mobilenet_v3 import MobileNetV3Large, MobileNetV3Small
from .resnet import (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
                     ResNet20CIFAR, ResNet32CIFAR, ResNet44CIFAR, ResNet56CIFAR, ResNetHyper,
                     ResNet18NCHW, ResNet34NCHW, ResNet50NCHW, ResNet101NCHW, ResNet152NCHW)
from .squeezenet import SqueezeNet
from .tiny_darknet import TinyDarknet
from .vgg import VGG16
from .nasnet import NASNetLarge, NASNetMobile, NASNetCIFAR
from .efficientnet import EfficientNetB0NCHW
from .hypermodels import SimpleHyper
from .fbnet_mobilenet import FBNet_MobileNetV2Imagenet, FBNet_MobileNetV2CIFAR, FBNet_MobileNetV2CIFARUP
from .pdart import Pdart


# to prevent Keras to bug for too big models.
# for instance ResNet152 with type2 does not work without this
sys.setrecursionlimit(10000)

model_name_to_class = {
    "AlexNet": AlexNet,
    "AlexNetQ": AlexNetQ,
    "AlexNetToy": AlexNetToy,
    "EfficientNetB0NCHW": EfficientNetB0NCHW,
    "MobileNetV2": MobileNetV2,
    "MobileNetV2Cifar10": MobileNetV2Cifar10,
    "MobileNetV2Cifar10_2": MobileNetV2Cifar10_2,
    "NASNetCIFAR": NASNetCIFAR,
    "NASNetLarge": NASNetLarge,
    "NASNetMobile": NASNetMobile,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "ResNet20CIFAR": ResNet20CIFAR,
    "ResNet32CIFAR": ResNet32CIFAR,
    "ResNet44CIFAR": ResNet44CIFAR,
    "ResNet56CIFAR": ResNet56CIFAR,
    "SqueezeNet": SqueezeNet,
    "TinyDarknet": TinyDarknet,
    "VGG16": VGG16,
    # Pdart model
    "Pdart": Pdart,
    # Channel first models
    "AlexNetNCHW": AlexNetNCHW,
    "ResNet18NCHW": ResNet18NCHW,
    "ResNet34NCHW": ResNet34NCHW,
    "ResNet50NCHW": ResNet50NCHW,
    "ResNet101NCHW": ResNet101NCHW,
    "ResNet152NCHW": ResNet152NCHW,
    "MobileNetV2NCHW": MobileNetV2NCHW,
    "MobileNetV2Cifar10NCHW": MobileNetV2Cifar10NCHW,
    "MobileNetV3Large": MobileNetV3Large,
    "MobileNetV3Small": MobileNetV3Small,
    # Hyper Model
    "SimpleHyper": SimpleHyper,
    "ResNetHyper": ResNetHyper,
    "MobileNetV2Cifar10Hyper": MobileNetV2Cifar10Hyper,
    # Architecture Search models
    "FBNet_MobileNetV2Imagenet": FBNet_MobileNetV2Imagenet,
    "FBNet_MobileNetV2CIFAR": FBNet_MobileNetV2CIFAR,
    "FBNet_MobileNetV2CIFARUP": FBNet_MobileNetV2CIFARUP,
}