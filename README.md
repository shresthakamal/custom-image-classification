 lglut-classify

## Build Instructions

Make sure the folders `A` and `U` containing images are place inside `data` folder like shown below:

```
       ROOT
        ├── api
        │   ├── app.py
        │   ├── routes.py
        │   ├── services
        │   │   └── predictor.py
        │   ├── static
        │   │   └── banner.jpg
        │   └── templates
        │       ├── index.html
        │       └── predict.html
        ├── data
        │   ├── A
        │   │   ├── A8.jpeg
        │   │   └── A9.jpeg
        │   ├── data.csv
        │   └── U
        │       ├── U8.jpeg
        │       └── U9.jpeg
        ├── lglutide
        │   ├── architectures
        │   │   ├── alexnet.py
        │   │   ├── cnn.py
        │   │   ├── densenet.py
        │   │   ├── nn.py
        │   │   └── resnet.py
        │   ├── dispatcher.py
        │   ├── logs
        │   ├── main.py
        │   ├── make_data.py
        │   ├── models
        │   │   └── 8
        │   │       ├── config.json
        │   │       └── Mar25_08_30_fold_4.pth
        │   ├── predict.py
        │   ├── runs
        │   └── utils
        │       ├── augmentations.py
        │       ├── logger.py
        │       └── options.py
        ├── Makefile
        ├── notebooks
        │   └── playground.ipynb
        ├── poetry.lock
        ├── pyproject.toml
        ├── README.md
        ├── requirements.txt
        └── tests
            └── test_lglutide.py

```

## Install Dependencies:

Before installing the project dependencies, make sure you are working on a virtual environment (conda/venv).

```python
make install
```
or
```python
pip install -r requirements.txt
```

## Training the model

All the training arguments and hyperparameters are specified in the `lglutide/config` module.

```python
python3 -m lglutide.main --experiment <experiment_no> --epochs <EPOCHS> --batch <BATCH>
```
eg:
```python
python3 -m lglutide.main --experiment 6 --epochs 1 --fold 2 --batch 64
```
**Training Flags**:

- Seed (`int`)
- Training Batch Size (`int`)
- Training Epochs (`int`)
- Image Channel (`int`)
- Image Width (`int`)
- Image Height (`int`)
- Learning Rate (`float`)
- Weight Decay Value  (`float`)
- Gradient Accumulation (`int`)
- Dropout (`float`)
- Number of k-folds (`int`)
- Experiment No (`int`)
- Data Augmentation (`bool`)
- Usage of GPU (`bool`)
- `--densenet`(default) or `--resnet` to specify training model  (`bool`)

Experimental Tracking sheet, [here](https://docs.google.com/spreadsheets/d/1DmFIhJwqj8ycNwWOrjpQC0-0WqSbJ-j2mNQz9H3F-Zc/edit?usp=sharing).

Checkpoints are uploaded [here](https://1drv.ms/f/s!Aprh41uH8yH1gcgVAU4c6iVMpqxP2Q?e=W6YBo3).


## Evaluating the Training metrics

The training logs like training loss, validation/test loss, F1 scores for individual epochs are all recorded using `tensorboard` writers. All the event logs are recorded under `runs/<experiment_no>`. To view the metrics, execute the following command:

```python
tensorboard --logdir=lglutide/runs/<experiment_no>
```
eg:
```python
tensorboard --logdir=lglutide/runs/6
```


## Inference

All the inferences are made through `config.json` files which contains the details about the trained model checkpoints, hyperparameters used to train the model, model names and more.

The `config.json` file generated after each training can be found in: ```lglutide/models/<experiment_no>/config.json```, depending on the experiment that we ran.

eg:
```lglutide/models/6/config.json```



### 1. Running Inference through CLI

- To run inference from command line MAKE SURE you have the following architecture:
    ```
      ROOT
        └──lglutide
            └── models
                └── <experiment_no>         # Training Experiment No
                    ├── config.json         # Training Hyperamaters
                    └── <saved_model>.pth   # Trained model checkpoint
    ```

-   Download the checkpoints corresponding to a specific experiment and run inference.

- Run inference using the following command:

```python
python3 -m lglutide.predict --config <CONFIG FILE PATH>
```
eg:
```python
python3 -m lglutide.predict --config "lglutide/models/6/config.json"
```
- This will prompt you to type the image path, **remeber the root foler before specifying the path**, eg: `data/A/A1.jpeg`


### 2. Running Inference through APP

- To run the inference through UI application, make sure you specify the `config.json` file on **`.env`** file (found in root directory) like shown below:
```python
CONFIG="lglutide/models/<experiment_no>/config.json"
```
eg:
```python
CONFIG="lglutide/models/6/config.json"
```


- Now execute the following command to start the application:

```python
python3 -m api.app
```

- You can now open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) on your browser to make the inferences
