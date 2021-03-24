import sys
import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from .alexnet import AlexNet, AlexNetQ, AlexNetToy
from .mobilenet import MobileNetV2, MobileNetV2Cifar10, MobileNetV2Cifar10_2, MobileNetV2Cifar10Hyper
from .mobilenet_v3 import MobileNetV3Large, MobileNetV3Small, MobileNetV3LargeCIFAR, MobileNetV3SmallCIFAR

from .resnet import (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
                     ResNet20CIFAR, ResNet32CIFAR, ResNet44CIFAR, ResNet56CIFAR, ResNetHyper)
from .wide_resnet import WideResNet28_10, WideResNet40_2
from .squeezenet import SqueezeNet
from .tiny_darknet import TinyDarknet
from .vgg import VGG16
from .nasnet import NASNetLarge, NASNetMobile, NASNetCIFAR
from .efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from .hypermodels import SimpleHyper
from .fbnet_mobilenet import FBNet_MobileNetV2Imagenet, FBNet_MobileNetV2CIFAR, FBNet_MobileNetV2CIFARUP
from .pdart import PdartsCIFAR, PdartsImageNet
from .complexnet import ShallowComplexNet, DeepComplexNet, WSComplexNetTF, WSComplexNetUpStride, DNComplexNetTF, DNComplexNetUpStride, IBComplexNetTF, IBComplexNetUpStride


# to prevent Keras to bug for too big models.
# for instance ResNet152 with type2 does not work without this
sys.setrecursionlimit(10000)

model_name_to_class = {
    "AlexNet": AlexNet,
    "AlexNetQ": AlexNetQ,
    "AlexNetToy": AlexNetToy,
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB1": EfficientNetB1,
    "EfficientNetB2": EfficientNetB2,
    "EfficientNetB3": EfficientNetB3,
    "EfficientNetB4": EfficientNetB4,
    "EfficientNetB5": EfficientNetB5,
    "EfficientNetB6": EfficientNetB6,
    "EfficientNetB7": EfficientNetB7,
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
    "WideResNet28_10": WideResNet28_10,
    "WideResNet40_2": WideResNet40_2,
    "SqueezeNet": SqueezeNet,
    "TinyDarknet": TinyDarknet,
    "VGG16": VGG16,
    "MobileNetV3Large": MobileNetV3Large,
    "MobileNetV3Small": MobileNetV3Small,
    "MobileNetV3LargeCIFAR": MobileNetV3LargeCIFAR,
    "MobileNetV3SmallCIFAR": MobileNetV3SmallCIFAR,
    # Pdart model
    "PdartsCIFAR": PdartsCIFAR,
    "PdartsImageNet": PdartsImageNet,
    # FIXME Below commented models have stale code and needs refactoring when prioritized.
    # # Hyper Model
    # "SimpleHyper": SimpleHyper,
    # "ResNetHyper": ResNetHyper,
    # "MobileNetV2Cifar10Hyper": MobileNetV2Cifar10Hyper,
    # # Architecture Search models
    # "FBNet_MobileNetV2Imagenet": FBNet_MobileNetV2Imagenet,
    # "FBNet_MobileNetV2CIFAR": FBNet_MobileNetV2CIFAR,
    # "FBNet_MobileNetV2CIFARUP": FBNet_MobileNetV2CIFARUP,
    # complexnet
    "ShallowComplexNet": ShallowComplexNet,
    "DeepComplexNet": DeepComplexNet,
    "WSComplexNetTF": WSComplexNetTF,
    "WSComplexNetUpStride": WSComplexNetUpStride,
    "DNComplexNetTF": DNComplexNetTF,
    "DNComplexNetUpStride": DNComplexNetUpStride,
    "IBComplexNetTF": IBComplexNetTF,
    "IBComplexNetUpStride": IBComplexNetUpStride,
}
