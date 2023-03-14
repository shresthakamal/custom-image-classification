import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

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

    for epoch in range(config.EPOCHS):  # loop over the dataset multiple times
        model.train()

        print(f"Epoch {epoch+1}\n-------------------------------")

        running_loss = 0.0
        training_stats[epoch] = {}

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

            # gradient accumulation in every 10 mini batch iterations
            if i % 15 == 0:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            if i % 50 == 0:  # print every 4 mini-batches
                print("[{}, {}] loss: {}".format(epoch + 1, i + 1, running_loss / 50))

                training_stats[epoch]["loss"] = running_loss / 50

                running_loss = 0.0

        # Model Evaluation
        model.eval()
        correct = 0
        total = 0

        true, preds = [], []

        with torch.no_grad():
            for data in testloader:
                images, labels = data

                images = images.reshape(
                    -1, config.IMAGE_C, config.IMAGE_H, config.IMAGE_W
                )
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

                # append true and preds to list by converting them to numpy arrays and detaching them
                true.append(labels.detach().cpu().numpy())
                preds.append(predicted.detach().cpu().numpy())

        print(
            "Accuracy of the network on the test images: %d %%\n"
            % (100 * correct / total)
        )

        # calculate F1 score based on true and predicted labels
        true = torch.cat(true)
        preds = torch.cat(preds)

        # print the F1 score
        print(f"""F1 Score: {f1_score(true, preds, average="weighted")}\n""")

        training_stats[epoch]["accuracy"] = 100 * correct / total

        # save the training stats to a file
        with open("lglutide/training_stats.txt", "a") as f:
            f.write(str(training_stats))

        # save the model
        torch.save(model.state_dict(), "lglutide/models/model.pth")

    print("Finished Training")
