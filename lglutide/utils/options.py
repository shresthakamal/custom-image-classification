import argparse

from lglutide.dispatcher import MODELS


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-seed", "--seed", help="Seed", required=False, default=1, type=int
    )
    parser.add_argument(
        "-batch",
        "--batch",
        help="Training Batch Size",
        required=False,
        default=64,
        type=int,
    )
    parser.add_argument(
        "-epochs",
        "--epochs",
        help="Number of training epochs",
        required=False,
        default=3,
        type=int,
    )
    parser.add_argument(
        "-channel",
        "--channel",
        help="Image Channel",
        required=False,
        default=3,
        type=int,
    )
    parser.add_argument(
        "-width", "--width", help="Image Width", required=False, default=256, type=int
    )
    parser.add_argument(
        "-height",
        "--height",
        help="Image Height",
        required=False,
        default=256,
        type=int,
    )
    parser.add_argument(
        "-lr", "--lr", help="Learning Rate", required=False, default=0.0001, type=int
    )
    parser.add_argument(
        "-decay",
        "--decay",
        help="Weight Decay Value",
        required=False,
        default=0.01,
        type=int,
    )
    parser.add_argument(
        "-grad_accumulate",
        "--grad_accumulate",
        help="Gradient Accumulation",
        required=False,
        default=10,
        type=int,
    )
    parser.add_argument(
        "-dropout", "--dropout", help="Dropout", required=False, default=0.2, type=int
    )
    parser.add_argument(
        "-fold", "--fold", help="K Folds", required=False, default=3, type=int
    )
    parser.add_argument(
        "-experiment",
        "--experiment",
        help="Experiment No",
        default=100,
        required=False,
        type=int,
    )
    parser.add_argument(
        "-aug",
        "--aug",
        help="Data Augmentation",
        required=False,
        default=False,
        type=bool,
    )
    parser.add_argument(
        "-download",
        "--download",
        help="Download the pre-trained weights",
        required=False,
        default=False,
        type=bool,
    )
    parser.add_argument(
        "-gpu", "--gpu", help="use GPU", required=False, default=0, type=str
    )
    parser.add_argument(
        "-densenet", "--densenet", help="Select Model DenseNet121", action="store_true"
    )
    parser.add_argument(
        "-resnet", "--resnet", help="Select Model Resnet50", action="store_true"
    )
    parser.add_argument(
        "-config", "--config", help="Config", required=False, default="None", type=str
    )
    args = vars(parser.parse_args())

    if args["resnet"]:
        args["model_name"] = "resnet"
        args["model"] = MODELS["resnet"]["model"]
        args["model_params"] = MODELS["resnet"]["params"]
    else:
        args["densenet"] = True
        args["model_name"] = "densenet"
        args["model"] = MODELS["densenet"]["model"]
        args["model_params"] = MODELS["densenet"]["params"]

    return args
