
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.dense1 = nn.Linear(4, 2)
        self.dense2 = nn.Linear(2, 4)

    def forward(self, x, return_bottlenecks: bool=False):
        bottlenecks = torch.sigmoid(self.dense1(x))
        outputs = torch.sigmoid(self.dense2(bottlenecks))

        if return_bottlenecks:
            return bottlenecks

        return outputs



