import io
import json
import os

import flask
import numpy as np
from dotenv import load_dotenv
from flask import render_template, request
from PIL import Image
from torchvision.transforms import ToTensor

from lglutide.predict import predict as engine_predict
from lglutide.utils.logger import logger


def home():
    return render_template(
        "index.html"
    )  # index.html is in api/templates/index.html and automatically inferred


def predict():
    load_dotenv()

    # get the uploaded file from the index.html form
    uploaded_file = request.files["file"]

    # convert the uploaded file to a PIL image
    image = Image.open(uploaded_file)

    # convert the PIL image to a tensor
    image = ToTensor()(image)

    # Error handling if the config file is not found
    try:
        config = os.getenv("CONFIG")
        logger.info(f"config: {config}")

        # load .json file as a dictionary
        with open(config, "r") as f:
            args = json.load(f)

    except Exception as e:
        return render_template(
            "predict.html",
            image=uploaded_file.filename,
            U=0,
            A=0,
            time_taken=0,
            error_message=e,
        )

    # get the prediction
    probas, time_taken = engine_predict(image, args)

    logger.info(f"Prediction: {probas}, Time Taken: {time_taken}")

    return render_template(
        "predict.html",
        image=uploaded_file.filename,
        U=round(probas[0] * 100, 3),
        A=round(probas[1] * 100, 3),
        time_taken=round(time_taken, 3),
    )
