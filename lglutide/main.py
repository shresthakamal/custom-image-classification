import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from lglutide.make_data import make_data
from lglutide.utils.logger import logger
from lglutide.utils.options import argument_parser


def train(**kwargs):
    # get the current time
    start = datetime.now().strftime("%b%d_%H:%M")

    # set random seed for reproducibility
    torch.manual_seed(kwargs["seed"])
    np.random.seed(kwargs["seed"])

    if kwargs["gpu"] and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    model = kwargs["model"](**kwargs["model_params"])
    model.to(device)
    logger.info(
        f"Number of model parameters: {sum(p.numel() for p in model.parameters())/1000000} M"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["decay"]
    )

    dataset = make_data(**kwargs)
    skf = StratifiedKFold(
        n_splits=kwargs["fold"], shuffle=True, random_state=kwargs["seed"]
    )

    logger.info("Training Started ...")

    training_stats = {}

    for fold, (train_ids, test_ids) in enumerate(
        skf.split(dataset, dataset.img_labels)
    ):
        logger.info(f"Fold: {fold}")
        logger.info(f"========")

        training_stats[fold] = {}

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=kwargs["batch"], sampler=train_subsampler
        )
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=kwargs["batch"], sampler=test_subsampler
        )

        # set tensorboard writer to save at runs directory based on folds and epochs and time
        writer = SummaryWriter(
            f"""lglutide/runs/{kwargs["experiment"]}/{start}/fold_{fold}"""
        )

        average_accuracy = 0
        average_f1score = 0

        for epoch in range(kwargs["epochs"]):  # loop over the dataset multiple times
            logger.info("---Epoch {}/{}---".format(epoch + 1, kwargs["epochs"]))

            model.zero_grad()

            # MODEL TRAINING
            model.train()
            running_loss = 0.0
            training_loss = []

            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # reshape the inputs to 4D tensor
                inputs = inputs.reshape(
                    -1, kwargs["channel"], kwargs["width"], kwargs["height"]
                )

                # zero the parameter gradients
                optimizer.zero_grad()

                # move the inputs and labels to the device
                inputs, labels = inputs.to(device), labels.to(device)

                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                training_loss.append(loss.item())

                # running_loss += loss.item()

                # # gradient accumulation in every 10 mini batch iterations
                # if i % kwargs["grad_accumulate"] == 0:
                loss.backward()
                optimizer.step()

                logger.info("[{}, {}] loss: {}".format(epoch + 1, i + 1, loss.item()))

                # plot the training loss only per fold per epoch
                writer.add_scalar(
                    "Training Loss", loss.item(), epoch * len(trainloader) + i
                )

                # running_loss = 0.0

            # MODEL EVALUATION
            model.eval()
            correct = 0
            total = 0
            true, preds = [], []
            testing_loss = []

            with torch.no_grad():
                for data in testloader:
                    images, labels = data

                    images = images.reshape(
                        -1, kwargs["channel"], kwargs["height"], kwargs["width"]
                    )
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    testing_loss.append(loss.item())

                    # plot the testing loss only as well
                    writer.add_scalar(
                        "Testing Loss", loss.item(), epoch * len(testloader) + i
                    )

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)

                    correct += (predicted == labels).sum().item()

                    # append true and preds to list by converting them to numpy arrays and detaching them
                    true.append(labels.detach().cpu().numpy())
                    preds.append(predicted.detach().cpu().numpy())

            # plot the mean training loss and testing loss per epoch in the same graph
            writer.add_scalars(
                "Loss",
                {
                    "Training Loss": np.mean(training_loss),
                    "Testing Loss": np.mean(testing_loss),
                },
                epoch,
            )

            # combine individual arrays inside the list to a single array
            true = np.concatenate(true)
            preds = np.concatenate(preds)

            accuracy = 100 * correct / total
            # print the accuracy of the network
            logger.info("Accuracy PE: %d %%" % (100 * correct / total))
            writer.add_scalar("Accuracy", accuracy, epoch)

            f1 = f1_score(true, preds, average="weighted")
            # print the F1 score of the network
            logger.info(f"""F1 Score PE: {f1}""")
            writer.add_scalar("F1 Score", f1, epoch)

            average_accuracy += accuracy
            average_f1score += f1

        exp = Path(f"""lglutide/models/{kwargs["experiment"]}""")

        if not os.path.exists(exp):
            os.makedirs(exp)

        torch.save(model.state_dict(), Path(exp, f"{start}_fold_{fold}.pth"))

        training_stats[fold]["accuracy"] = average_accuracy / kwargs["epochs"]
        training_stats[fold]["f1_score"] = average_f1score / kwargs["epochs"]

    writer.close()

    config_path = Path(exp, f"""config.json""")
    with open(config_path, "w") as f:
        del kwargs["model"]
        kwargs["checkpoint"] = Path(exp, f"{start}_fold_{fold}.pth" "").as_posix()
        kwargs["config"] = config_path.as_posix()
        json.dump(kwargs, f, indent=4)

    # create a dataframe to store the training stats
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.transpose()
    df_stats.to_csv("lglutide/training_stats.csv", index=False)

    logger.info(f"Training Stats:\n{df_stats}")

    logger.info("Finished Training")


def run():
    args = argument_parser()

    logger.info(args)

    # set CUDA VISIBLE DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]

    train(**args)


if __name__ == "__main__":
    run()
