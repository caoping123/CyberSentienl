import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import *
from model import M5
from mfcc import mfcc
import csv

train_loader, test_loader = data_loader(1024)
train_set = SubsetSC("training")
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)
# 1 35
model = M5(n_input=transformed.shape[0], n_output=len(labels))
model.to(device)
model.load_state_dict(torch.load('./stress/model//model_116.pth'))
criterion = torch.nn.NLLLoss()
regularization = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

m1 = mfcc('./voice/1.wav')
m1 = torch.mean(m1, dim=1).reshape(-1, 1).to(device)  # 13, 1
m2 = mfcc('./voice/2.wav')
m2 = torch.mean(m2, dim=1).reshape(-1, 1).to(device)
m3 = mfcc('./voice/3.wav')
m3 = torch.mean(m3, dim=1).reshape(-1, 1).to(device)

X1 = torch.load('./stress/projection_1.pth').to(device)  # 13, 96
X2 = torch.load('./stress/projection_2.pth').to(device)  # 13, 96
X3 = torch.load('./stress/projection_3.pth').to(device)  # 13, 96


def train(model, epoch, log_interval):
    model.train()
    losses = []
    lw1_loss = []
    lw2_loss = []
    lw3_loss = []

    for data, target in tqdm(train_loader, desc='Training', unit='batch', leave=False):
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        lt = criterion(output.squeeze(), target)

        conv2_channel = [5, 7, 8, 10, 13, 17, 19, 23, 27, 30]
        conv3_channel = [0, 11, 13, 14, 18, 22, 23, 24, 29, 30, 31, 34, 35, 37, 40, 43, 47, 49, 51, 54, 56, 57, 61]
        W = torch.stack([model.conv2.weight[i] for i in conv2_channel] + [model.conv3.weight[i] for i in conv3_channel],
                        dim=0)
        w1 = torch.flatten(torch.mean(W[:11], dim=0)).reshape(-1, 1)  # 96, 1
        w2 = torch.flatten(torch.mean(W[11:22], dim=0)).reshape(-1, 1)  # 96, 1
        w3 = torch.flatten(torch.mean(W[22:], dim=0)).reshape(-1, 1)  # 96, 1

        lw1 = regularization(torch.mm(X1, w1), m1)
        lw2 = regularization(torch.mm(X2, w2), m2)
        lw3 = regularization(torch.mm(X3, w3), m3)

        loss = lt + lw1 + lw2 + lw3

        optimizer.zero_grad()
        loss.backward()
        # lw.backward()
        optimizer.step()

        # print training stats
        # if batch_idx % log_interval == 0:
        #     print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        # record loss
        losses.append(lt.item())
        lw1_loss.append(lw1.item())
        lw2_loss.append(lw2.item())
        lw3_loss.append(lw3.item())

    print(f"Train Epoch: {epoch}\tLoss: {sum(losses) / len(losses):.6f}\tLw1: {sum(lw1_loss) / len(lw1_loss):.6f}"
          f"\tLw2: {sum(lw2_loss) / len(lw2_loss):.6f}\tLw3: {sum(lw3_loss) / len(lw3_loss):.6f}")
    epoch_ret.append(sum(lw1_loss) / len(lw1_loss))
    epoch_ret.append(sum(lw2_loss) / len(lw2_loss))
    epoch_ret.append(sum(lw3_loss) / len(lw3_loss))


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        # pbar.update(pbar_update)

    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
    epoch_ret.append(100. * correct / len(test_loader.dataset))


if __name__ == '__main__':
    log_interval = 20
    n_epoch = 200
    # losses = []

    # The transform needs to live on the same device as the model and the data.
    transform = transform.to(device)
    for epoch in range(117, n_epoch + 1):
        epoch_ret = []
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()
        # if epoch % 10 == 0:

        torch.save(model.state_dict(), 'stress/model/model_'+str(epoch)+'.pth')

        csv_file = './stress/embd.csv'
        with open(csv_file, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([epoch_ret])
