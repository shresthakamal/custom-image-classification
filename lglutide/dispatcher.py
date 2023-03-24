from lglutide.architectures.densenet import DenseNet121
from lglutide.architectures.resnet import resnet34

MODELS = {
    "densenet": {
        "model": DenseNet121,
        "params": {"num_classes": 2, "grayscale": False},
    },
    "resnet": {"model": resnet34, "params": {"num_classes": 2}},
}
