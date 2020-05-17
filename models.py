"""Differents models used for Project1"""

import torch
from torch import nn
from torch.nn import functional as F
import torchsummary

class Net1(nn.Module):
    """First model used for the project, the simplest one.

        This first model uses no weight sharing, the two input images are passed
        through two separate convolutional networks. The output of these 2
        networks are then concatenated and passed through 4 fully connected
        layers. There is a single output neuron indicating if the first digit
        is larger (output should be 1) or if the second one is (output should be
        0)."""

    def __init__(self):
        super(Net1,self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv2_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.fc1 = nn.Linear(32*5*5*2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128,1)

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
        x = self.max_pool(F.relu(self.conv1_2(x)))
        x = x.view(x.size(0), -1)
        y = F.relu(self.conv2_1(img2))
        y = self.max_pool(F.relu(self.conv2_2(y)))
        y = y.view(y.size(0), -1)
        z = torch.cat((x,y), 1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = torch.sigmoid(self.fc4(z))
        return z


class Net2(nn.Module):
    """Second model, introduces weight sharing for the convolutional layers.

        This second model uses weight sharing, the two images are passed through
        a single convolutional network. The outputs are then concatenated and
        passed through 4 fully connected layers. There is a single output neuron
        indicating if the first digit is larger (output should be 1) or if the
        second one is (output should be 0).
    """

    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.fc1 = nn.Linear(32*5*5*2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128,1)

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
        x = self.max_pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        y = F.relu(self.conv1(img2))
        y = self.max_pool(F.relu(self.conv2(y)))
        y = y.view(y.size(0), -1)
        z = torch.cat((x,y), 1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = torch.sigmoid(self.fc4(z))
        return z


class Net3(nn.Module):
    """Third model, we use the label of the digits as an auxiliary loss.

        This third model introduces a set of new outputs. The network must now
        also predict the digit of each image. This is used as an auxiliary loss.
        The two images are passed through a single convolutional network. The
        outputs are concatenated and passed through a series of fully connected
        layers. The network then has 3 outputs:
            - A single neuron to predict which digit is the largest.
            - Two softmax outputs two predict the labels of the 2 images. These
                two outputs use weight sharing
    """

    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.fc1 = nn.Linear(32*5*5*2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128,1)
        self.fc5 = nn.Linear(128,10)

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
        x = self.max_pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        y = F.relu(self.conv1(img2))
        y = self.max_pool(F.relu(self.conv2(y)))
        y = y.view(y.size(0), -1)
        z = torch.cat((x,y), 1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        pred = torch.sigmoid(self.fc4(z))
        digit1 = F.log_softmax(self.fc5(z),dim=1)
        digit2 = F.log_softmax(self.fc5(z),dim=1)
        return torch.cat((pred, digit1, digit2), 1)

class Net4(nn.Module):
    """Third model, we use the label of the digits as an auxiliary loss.

        This fourth model adds batch normalisation on top of the thrid model.
        The network then has 3 outputs:
            - A single neuron to predict which digit is the largest.
            - Two softmax outputs two predict the labels of the 2 images. These
                two outputs use weight sharing
    """

    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.fc1 = nn.Linear(32*5*5*2, 32)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.fc2 = nn.Linear(32, 64)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.fc3 = nn.Linear(64, 128)
        self.bn5 = nn.BatchNorm1d(num_features=128)
        self.fc4 = nn.Linear(128,1)
        self.fc5 = nn.Linear(128,10)

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
        x = F.relu(self.bn1(self.conv1(img1)))
        x = self.max_pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        y = F.relu(self.bn1(self.conv1(img2)))
        y = self.max_pool(F.relu(self.bn2(self.conv2(y))))
        y = y.view(y.size(0), -1)
        z = torch.cat((x,y), 1)
        z = F.relu(self.bn3(self.fc1(z)))
        z = F.relu(self.bn4(self.fc2(z)))
        z = F.relu(self.bn5(self.fc3(z)))
        pred = torch.sigmoid(self.fc4(z))
        digit1 = F.log_softmax(self.fc5(z),dim=1)
        digit2 = F.log_softmax(self.fc5(z),dim=1)
        return torch.cat((pred, digit1, digit2), 1)

if __name__ == "__main__":
    input = torch.ones((100,2,14,14))
    print(input.size())
    net = Net4()
    print(net(input).size())
