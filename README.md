# lglut-classify

## Build Instructions

Make sure the folders `A` and `U` containing images are place inside `data` folder like shown below:

```
.
├── data                        # contains two folders: A, U containing images
│   ├── A
│   │   ├── A10.jpeg
│   │   ├── A1.jpeg
│   │   └──A2.jpeg
│   ├── test.csv
│   ├── train.csv
│   └── U
│       ├── U10.jpeg
│       ├── U1-1.jpeg
│       └── U1-2.jpeg
├── lglutide                    # Contains all the modules for training and testing
│   ├── config.py               # Arguments and Hyperparameters
│   ├── main.py                 # Training and Validation Pipeline
│   ├── make_data.py            # Prepare Data
│   ├── models                  # Saved Model Checkpoints
│   │   ├── model_0.pth
│   │   ├── model_1.pth
│   │   ├── model_2.pth
│   │   ├── model_3.pth
│   │   ├── model_4.pth
│   │   ├── model_5.pth
│   │   ├── model_6.pth
│   │   ├── model_7.pth
│   │   ├── model_8.pth
│   │   └── model_9.pth
│   ├── nn.py                   # Neural Network Architecture
│   ├── predict.py              # Inference Pipeline
├── Makefile                    # make commands for ease developments
├── poetry.lock
├── pyproject.toml              # Project Dependencies
├── README.md
├── requirements.txt
└── tests

```

## Install Dependencies:

Before installing the project dependencies, make sure you are working on a virtual environment (conda/venv).

```python
make install
```

## Training the model

All the training arguments and hyperparameters are specified in the `lglutide/config` module.

```python
make train
```

## Running Inference

- To run inference make sure all the saved model checkpoints are in `lglutide/models` folder

- Look into the `lglutide/config` file to check which checkpoint is being used for inference.

- run inference using the following command and select an image to run inference on. eg: `data/A/A1.jpeg`

```python
make inference
```
