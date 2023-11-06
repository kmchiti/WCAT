import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import Base_Model

__all__ = [
    "LeNet",
    "Simple_MLP"
]


class LeNet_base(nn.Module):
    # network structure
    def __init__(self, num_classes=10):
        super(LeNet_base, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        One forward pass through the network.

        Args:
            x: input
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """
        Get the number of features in a batch of tensors `x`.
        """
        size = x.size()[1:]
        return np.prod(size)


class Simple_MLP_base(nn.Module):
    def __init__(self, num_classes=10):
        super(Simple_MLP_base, self).__init__()

        self.layer1 = nn.Linear(784, 200)
        self.layer2 = nn.Linear(200, 50)
        self.layer3 = nn.Linear(50, num_classes)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.activation(self.layer1(x))
        out = self.activation(self.layer2(out))
        out = self.layer3(out)

        return out

@Base_Model.register('lenet5')
class LeNet(Base_Model, LeNet_base):
    def __init__(self, num_classes=10, pretrained=False, progress=True,
                 **kwargs):
        LeNet_base.__init__(self, num_classes=num_classes)
        Base_Model.__init__(self, **kwargs)

@Base_Model.register('simple_mlp')
class Simple_MLP(Base_Model, Simple_MLP_base):
    def __init__(self, num_classes=10, pretrained=False, progress=True,
                 **kwargs):
        Simple_MLP_base.__init__(self, num_classes=num_classes)
        Base_Model.__init__(self, **kwargs)
