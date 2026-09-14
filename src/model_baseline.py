import torch.nn as nn
from torchvision.models import resnet18

class BaselineResNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # bona fide = 0, fake = 1
        net.fc = nn.Linear(net.fc.in_features,2)
        self.net = net

    def forward(self, x):
        return self.net(x)