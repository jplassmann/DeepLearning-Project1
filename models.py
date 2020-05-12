"""Differents models used for Project1"""

import torch
from torch import nn
from torch.nn import functional as F


class Net1(nn.Module):
    """First model used for the project, the simplest one.

        This first model uses no weight sharing, the two input images are passed
        through two separate convolutional networks. The output of these 2
        networks are then concatenated and passed through 2 fully connected
        layers. There is a single output neuron indicating if the first digit
        is larger (output should be 1) or if the second one is (output should be
        0)."""

    def __init__(self):
        super(Net1,self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv2_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32*10*10*2, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, input):
        """Forward pass of the network

            Args:
            input: A (n, 2, 14, 14) pytorch tensor representing the images of
                the 2 digits.

            Returns:
                A (n,1) tensor representing the prediction of the largest of
                the two digits given as input.
        """
        img1 = torch.narrow(input,dim=1,start=0,length=1)
        img2 = torch.narrow(input,dim=1,start=1,length=1)
        x = F.relu(self.conv1_1(img1))
        x = F.relu(self.conv1_2(x))
        x = x.view(x.size(0), -1)
        y = F.relu(self.conv2_1(img2))
        y = F.relu(self.conv2_2(y))
        y = y.view(y.size(0), -1)
        z = torch.cat((x,y), 1)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        return z


class Net2(nn.Module):
    """Second model, introduces weight sharing for the convolutional layers.

        This second model uses weight sharing, the two images are passed through
        a single convolutional network. The outputs are then concatenated and
        passed through 2 fully connected layers. There is a single output neuron
        indicating if the first digit is larger (output should be 1) or if the
        second one is (output should be 0).
    """

    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32*10*10*2, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, input):
        """Forward pass of the network

            Args:
            input: A (n, 2, 14, 14) pytorch tensor representing the images of
                the 2 digits.

            Returns:
                A (n,1) tensor representing the prediction of the largest of
                the two digits given as input.
        """
        img1 = torch.narrow(input,dim=1,start=0,length=1)
        img2 = torch.narrow(input,dim=1,start=1,length=1)
        x = F.relu(self.conv1(img1))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        y = F.relu(self.conv1(img2))
        y = F.relu(self.conv2(y))
        y = y.view(y.size(0), -1)
        z = torch.cat((x,y), 1)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        return z


class Net3(nn.Module):
    """Third model, we use the label of the digits as an auxiliary loss.

        This third model introduces a set of new outputs. The network must now
        also predict the digit of each image. This is used as an auxiliary loss.
        The two images are passed through a single convolutional network. The
        outputs are concatenated and passed through a fully connected layer.
        There are then 3 outputs:
            - A single neuron to predict which digit is the largest.
            - Two softmax outputs two predict the labels of the 2 images. These
                two outputs use weight sharing
    """

    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32*10*10*2, 120)
        self.fc2 = nn.Linear(120, 1)
        self.fc3 = nn.Linear(120,10)

    def forward(self, input):
        """Forward pass of the network

            Args:
            input: A (n, 2, 14, 14) pytorch tensor representing the images of
                the 2 digits.

            Returns:
                A (n,21) tensor representing the prediction of the largest of
                the two digits given as input as well as a log-softmax of the
                predictions of the digits themselves.
        """
        img1 = torch.narrow(input,dim=1,start=0,length=1)
        img2 = torch.narrow(input,dim=1,start=1,length=1)
        x = F.relu(self.conv1(img1))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        y = F.relu(self.conv1(img2))
        y = F.relu(self.conv2(y))
        y = y.view(y.size(0), -1)
        z = torch.cat((x,y), 1)
        z = F.relu(self.fc1(z))
        pred = torch.sigmoid(self.fc2(z))
        digit1 = nn.LogSoftmax(self.fc3(z))
        digit2 = nn.LogSoftmax(self.fc3(z))
        return torch.cat((pred, digit1, digit2), 1)
