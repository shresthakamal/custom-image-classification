import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from lglutide import config
from lglutide.make_data import make_data
from lglutide.nn import NNModel

if __name__ == "__main__":
    # set random seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NNModel()
    model.to(device)
    print(
        f"Number of model parameters: {sum(p.numel() for p in model.parameters())/1000000} M"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    dataset = make_data()
    skf = StratifiedKFold(
        n_splits=config.k_fold, shuffle=True, random_state=config.SEED
    )

    print("Training Started ...\n")

    training_stats = {}

    for fold, (train_ids, test_ids) in enumerate(
        skf.split(dataset, dataset.img_labels)
    ):
        print(f"Fold: {fold}")
        print(f"========")

        training_stats[fold] = {}

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.BATCHSIZE, sampler=train_subsampler
        )
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.BATCHSIZE, sampler=test_subsampler
        )

        model.zero_grad()

        # set tensorboard writer to save at runs directory based on folds and epochs
        writer = SummaryWriter(f"lglutide/runs/fold_{fold}")

        average_accuracy = 0
        average_f1score = 0

        for epoch in range(config.EPOCHS):  # loop over the dataset multiple times
            print("---Epoch {}/{}---".format(epoch + 1, config.EPOCHS))

            # MODEL TRAINING
            model.train()
            running_loss = 0.0
            training_loss = []

            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # reshape the inputs to 4D tensor
                inputs = inputs.reshape(
                    -1, config.IMAGE_C, config.IMAGE_W, config.IMAGE_H
                )

                # zero the parameter gradients
                optimizer.zero_grad()

                # move the inputs and labels to the device
                inputs, labels = inputs.to(device), labels.to(device)

                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                training_loss.append(loss.item())

                running_loss += loss.item()

                # gradient accumulation in every 10 mini batch iterations
                if i % config.gradient_accumulate_per_mini_batch == 0:
                    loss.backward()
                    optimizer.step()

                # print loss in every 50 mini batch iterations
                if i % 50 == 0:
                    print(
                        "[{}, {}] loss: {}".format(epoch + 1, i + 1, running_loss / 50)
                    )

                    # plot the training loss only per fold
                    writer.add_scalar(
                        "Training Loss", running_loss / 50, epoch * len(trainloader) + i
                    )

                    running_loss = 0.0

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
                        -1, config.IMAGE_C, config.IMAGE_H, config.IMAGE_W
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

            # plot training and testing loss in the same figure
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
            print("\nAccuracy PE: %d %%" % (100 * correct / total))
            writer.add_scalar("Accuracy", accuracy, epoch)

            f1 = f1_score(true, preds, average="weighted")
            # print the F1 score of the network
            print(f"""F1 Score PE: {f1}\n""")
            writer.add_scalar("F1 Score", f1, epoch)

            average_accuracy += accuracy
            average_f1score = f1

            # save the model based on folds and epochs
            torch.save(model.state_dict(), f"lglutide/models/model_{fold}.pth")

        training_stats[fold]["accuracy"] = average_accuracy / config.EPOCHS
        training_stats[fold]["f1_score"] = average_f1score / config.EPOCHS

        writer.close()

    # create a dataframe to store the training stats
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.transpose()
    df_stats.to_csv("lglutide/training_stats.csv", index=False)

    print(df_stats)

    print("Finished Training")
