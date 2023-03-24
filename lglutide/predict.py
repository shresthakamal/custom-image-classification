import json
import time

import torch
import torchvision.transforms as transforms
from torchvision.io import read_image

from lglutide.dispatcher import MODELS
from lglutide.utils.logger import logger
from lglutide.utils.options import argument_parser


def check_if_model_exists(model_path, download=False):
    try:
        with open(model_path, "rb") as f:
            logger.info("Model found. Loading model...")
        return True
    except FileNotFoundError:
        if download:
            import gdown

            # download the pretrained model from google drive
            logger.info("Model not found. Downloading model...")
            gdown.download(
                "https://drive.google.com/uc?id=1Y8W5YV0vJZu5wZoJ8Zu5E6jv9e9X7B9O",
                model_path,
                quiet=False,
            )
        else:
            return False


def predict(image, args):
    # start the timer
    start = time.time()

    if check_if_model_exists(args["checkpoint"], download=args["download"]):
        if args["gpu"] and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        image = transforms.Resize((args["width"], args["height"]))(image)
        image = image.numpy()
        image = transforms.ToTensor()(image)
        image = image.reshape(-1, args["channel"], args["width"], args["height"])

        model = MODELS[args["model_name"]]["model"](**args["model_params"])
        model = model.to(device)

        try:
            model.load_state_dict(torch.load(args["checkpoint"]))
        except:
            raise Exception("Checkpoint not valid with selected model !!")

        image = image.to(device)

        # set the model to evaluation mode
        model.eval()

        # get the predictions
        with torch.no_grad():
            probas = model(image)

        # detach the tensor from the graph
        probas = probas.detach().cpu().numpy()[0]

        return probas, time.time() - start
    else:
        raise Exception(
            "Model not found. Please train your model or download the pretrained model."
        )


# python main block
if __name__ == "__main__":
    img_path = input("Enter the image path: ")
    logger.info(img_path)

    image = read_image(img_path)

    args = argument_parser()

    try:
        # load .json file as a dictionary
        with open(args["config"], "r") as f:
            args = json.load(f)
    except FileNotFoundError:
        raise Exception("Config file not found. Please check the path.")

    logger.info(f"(Prediction, Time Taken): {predict(image, args)}")
