import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import torch.optim as optim
import torchaudio
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


if __name__ == '__main__':
    model = M5()
    model.load_state_dict(torch.load("model.pth"))
    # conv2_w = []
    # conv3_w = []
    # for i in range(model.conv2.weight.data.shape[0]):
    #     conv2_w.append(torch.sum(torch.abs(model.conv2.weight.data[i])).item())
    #
    # for i in range(model.conv3.weight.data.shape[0]):
    #     conv3_w.append(torch.sum(torch.abs(model.conv3.weight.data[i])).item())

    # plt.bar(range(len(conv3_w)+len(conv2_w)), conv2_w + conv3_w, color='skyblue')
    # plt.title("total")
    # plt.show()


    # ret = [conv2_w, conv3_w]
    # csv_file = 'L1.cvs'
    # with open(csv_file, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(ret)

    # for index, i in enumerate(conv2_w):
    #     if i < 19:
    #         print(index, end=" ")
    #
    # print()
    # for index, i in enumerate(conv3_w):
    #     if i < 19:
    #         print(index, end=" ")

    # conv2_channel = [5, 7, 8, 10, 13, 17, 19, 23, 27, 30]
    # conv3_channel = [0, 11, 13, 14, 18, 22, 23, 24, 29, 30, 31, 34, 35, 37, 40, 43, 47, 49, 51, 54, 56, 57, 61]
    # W = torch.stack([model.conv2.weight[i] for i in conv2_channel] + [model.conv3.weight[i] for i in conv3_channel], dim=0)
    # w = torch.flatten(torch.mean(W, dim=0)).reshape(-1, 1)
