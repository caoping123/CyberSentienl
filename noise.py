import numpy
import torch
from model import M5
from mfcc import mfcc
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import librosa
from draw import env


def extract():
    model = M5()
    model.load_state_dict(torch.load('wm_model/model_43'))
    X = torch.load('projection.pth')

    conv2_channel = [5, 7, 8, 10, 13, 17, 19, 23, 27, 30]
    conv3_channel = [0, 11, 13, 14, 18, 22, 23, 24, 29, 30, 31, 34, 35, 37, 40, 43, 47, 49, 51, 54, 56, 57, 61]
    W = torch.stack([model.conv2.weight[i] for i in conv2_channel] + [model.conv3.weight[i] for i in conv3_channel],
                    dim=0)
    w = torch.flatten(torch.mean(W, dim=0)).reshape(-1, 1).data  # 96, 1
    return torch.mm(X, w)  # [13, 1]


def seg_wav(filepath):
    m = mfcc(filepath)
    ret =[]
    for i in range(int(m.shape[1] / 1000)):
        seg = m[:, i * 1000: i * 1000 + 1000]
        seg = torch.mean(seg, dim=1)
        ret.append(seg)
    return ret

def compare(filepath, wm):
    mfcc = seg_wav(filepath)
    # pc = []
    cos = []
    for seg in mfcc:
        # pc.append(numpy.corrcoef(seg.numpy(), m.numpy())[0, 1])
        cos.append(F.cosine_similarity(seg, m, dim=0).item())
    # print(pc)
    return cos


def get_mean(filepath):
    features = mfcc(filepath)
    features = torch.mean(features, dim=1)
    return features


def compare_features(lhs, rhs):
    return F.cosine_similarity(lhs, rhs, dim=0).item()


if __name__ == '__main__':
    m = extract().squeeze(1)
    # m = get_mean('./voice/env/distance/10cm.wav')
    print('距离10cm:\t', end='')
    a = compare('./voice/env/distance/10cm.wav', m)
    print(a)
    print('距离20cm:\t', end='')
    b = compare('./voice/env/distance/20cm.wav', m)
    print(b)
    print('距离40cm:\t', end='')
    c = compare('./voice/env/distance/40cm.wav', m)
    print(c)
    print('正对麦克风:\t', end='')
    d = compare('./voice/env/towards/front.wav', m)
    print(d)
    print('背对麦克风:\t', end='')
    e = compare('./voice/env/towards/back.wav', m)
    print(e)
    print('侧对麦克风:\t', end='')
    f = compare('./voice/env/towards/side.wav', m)
    print(f)
    print('固定频率噪声:\t', end='')
    g = compare('./voice/env/frequent/noise.wav', m)
    print(g)
    print('无固定噪声:\t', end='')
    h = compare('./voice/env/frequent/clean.wav', m)
    print(h)
    print('室外环境:\t', end='')
    i = compare('./voice/env/frequent/outside.wav', m)
    print(i)

    # env.draw(a, b, c, ['10cm', '20cm', '40cm'], 'draw/distance.pdf')
    # env.draw(d, e, f, ['front', 'back', 'side'], 'draw/towards.pdf')
    # env.draw(h, g, i, ['Noiseless', 'Fixed frequency noise', 'Outside'], 'draw/env.pdf')

    # env.line_chat(a, b, c, ['10cm', '20cm', '40cm'], 'draw/distance.pdf')

    #env.box_plots(a, b, c, ['25cm', '50cm', '100cm'], 'draw/distance.pdf')
    env.box_plots(h, g, i, ['Noiseless', 'Fixed noise', 'Outside'], 'draw/env.pdf')
    #env.box_plots(d, f, e, ['Front', 'Side', 'Back'], 'draw/towards.pdf')
