import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_1 = 512
        hidden_2 = 512
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 11)


    #         self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SimpleNet5(nn.Module):

    def __init__(self):
        super(SimpleNet5, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        # self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        # self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        # self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        # self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        # self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        # self.prelu3_2 = nn.PReLU()

        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128 * 3 * 3, 2)
        self.ip2 = nn.Linear(2, 10, bias=False)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        # x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        # x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        # x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 3 * 3)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)

        return ip1, ip2


# LeNet-5
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                                    nn.BatchNorm2d(6),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(4 * 4 * 16, 120),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84),
                                 nn.ReLU())
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
