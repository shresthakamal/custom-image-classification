# create a CNN model with 2 convolutional layers and 3 fully connected layers and output layer with 2 classes
# model = CNNModel()
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5)

        self.fc1 = nn.Linear(in_features=256 * 753 * 1005, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 256 * 753 * 1005)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x
