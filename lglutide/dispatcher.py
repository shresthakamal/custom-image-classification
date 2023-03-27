from lglutide.architectures.alexnet import AlexNet
from lglutide.architectures.cnn import ConvNet
from lglutide.architectures.densenet import DenseNet121
from lglutide.architectures.nn import NNModel
from lglutide.architectures.resnet import resnet34
from lglutide.architectures.vgg import VGG16

MODELS = {
    "densenet": {
        "model": DenseNet121,
        "params": {"num_classes": 2, "grayscale": False},
    },
    "resnet": {"model": resnet34, "params": {"num_classes": 2}},
    "vgg": {
        "model": VGG16,
        "params": {
            "num_features": 3,
            "num_classes": 2,
        },
    },
    "cnn": {
        "model": ConvNet,
        "params": {
            "num_classes": 2,
        },
    },
    "alexnet": {
        "model": AlexNet,
        "params": {
            "num_classes": 2,
        },
    },
    "nn": {"model": NNModel, "params": {}},
}
