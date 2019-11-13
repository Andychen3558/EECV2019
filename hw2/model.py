import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(0.25),  # [6, 14, 14]

            nn.Conv2d(6, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(0.3)  # [16, 7, 7]
        )
        self.fc = nn.Sequential(
            nn.Linear(16*7*7, 120),
            nn.BatchNorm1d(120),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # TODO
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def name(self):
        return "ConvNet"


class Fully(nn.Module):
    def __init__(self):
        super(Fully, self).__init__()
        # TODO
        self.fc = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0),-1) # flatten input tensor
        # TODO
        out = self.fc(x)
        return out

    def name(self):
        return "Fully"

