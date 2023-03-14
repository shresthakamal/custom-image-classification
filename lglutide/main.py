import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

from lglutide import config
from lglutide.make_data import make_data
from lglutide.nn import NNModel

if __name__ == "__main__":
    # set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = make_data()

    model = NNModel()

    # print the number of model parameters
    print(
        f"Number of model parameters: {sum(p.numel() for p in model.parameters())/1000000} M"
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM
    )

    print("Training Started ...\n")

    model.zero_grad()

    training_stats = {}

    # initialize tensorboard to save at runs directory
    writer = SummaryWriter("lglutide/runs")

    for epoch in range(config.EPOCHS):  # loop over the dataset multiple times
        model.train()

        print(f"Epoch {epoch+1}\n-------------------------------")

        running_loss = 0.0
        training_loss = []

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # reshape the inputs to 4D tensor
            inputs = inputs.reshape(-1, config.IMAGE_C, config.IMAGE_W, config.IMAGE_H)

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
            # print every 10 mini-batches
            if i % 10 == 0:
                loss.backward()
                optimizer.step()

                print("[{}, {}] loss: {}".format(epoch + 1, i + 1, running_loss / 10))

                # plot the training loss only as well
                writer.add_scalar(
                    "Training Loss", running_loss / 10, epoch * len(trainloader) + i
                )

                running_loss = 0.0

        # Model Evaluation
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

        # print the accuracy of the network
        print(
            "\nAccuracy of the network on the test images: %d %%"
            % (100 * correct / total)
        )
        writer.add_scalar("Accuracy", 100 * correct / total, epoch)

        # print the F1 score
        print(f"""F1 Score: {f1_score(true, preds, average="weighted")}\n""")
        writer.add_scalar("F1 Score", f1_score(true, preds, average="weighted"), epoch)

        # save the model
        torch.save(model.state_dict(), f"lglutide/models/model_{epoch}.pth")

    print("Finished Training")
