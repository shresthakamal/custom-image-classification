import io

import flask
import numpy as np
from flask import render_template, request
from PIL import Image
from torchvision.transforms import ToTensor

from lglutide.nn import NNModel
from lglutide.predict import predict as engine_predict


def home():
    return render_template(
        "index.html"
    )  # index.html is in api/templates/index.html and automatically inferred


def predict():
    # get the uploaded file from the index.html form
    uploaded_file = request.files["file"]

    # convert the uploaded file to a PIL image
    image = Image.open(uploaded_file)

    # convert the PIL image to a tensor
    image = ToTensor()(image)

    # get the prediction
    probas, time_taken = engine_predict(image)

    return render_template(
        "predict.html",
        image=uploaded_file.filename,
        U=round(probas[0] * 100, 2),
        A=round(probas[1] * 100, 2),
        time_taken=time_taken,
    )
