import torch
import torch.nn as nn
import torch.optim as optim

from lglutide.cnn import CNNModel
from lglutide.make_data import make_data

if __name__ == "__main__":
    # set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # device = "cpu"

    trainloader, testloader = make_data()

    # create a CNN model with 2 convolutional layers and 3 fully connected layers and output layer with 2 classes
    model = CNNModel()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()

    model.zero_grad()

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # reshape the inputs to 4D tensor
            inputs = inputs.reshape(-1, 3, 3024, 4032)

            # zero the parameter gradients
            optimizer.zero_grad()

            # move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            print(loss)

    print("Finished Training")
