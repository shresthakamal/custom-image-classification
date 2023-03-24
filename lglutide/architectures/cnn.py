import torch
import torch.nn as nn
import torch.nn.functional as F

from lglutide import config


# define a convolutional neural network
class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        # batch x 3 x 256 x 256 -> batch x 8 x 256 x 256
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
        )

        # batch x 8 x 28 x 28 -> batch x 8 x 128 x 128
        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)
        )

        # batch x 8 x 14 x 14 -> batch x 16 x 14 x 14
        self.conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
        )

        # batch x 16 x 14 x 14 -> batch x 16 x 7 x 7
        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)
        )

        self.linear1 = torch.nn.Linear(256 * 64 * 64, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)

        logits = self.linear1(out.view(-1, 256 * 64 * 64))

        probas = F.softmax(logits, dim=1)

        return probas
