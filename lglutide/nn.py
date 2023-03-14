import torch.nn as nn
import torch.nn.functional as F

from lglutide import config


class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(config.IMAGE_C * config.IMAGE_W * config.IMAGE_H, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        # softmax layer
        probas = F.softmax(logits, dim=1)

        return probas
