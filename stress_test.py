import numpy as np
import torch
from model import M5
from mfcc import mfcc
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv


def extract():
    model = M5()
    model.load_state_dict(torch.load('stress/model/model_134.pth'))
    X1 = torch.load('./stress/projection_1.pth') # 13, 96
    X2 = torch.load('./stress/projection_2.pth') # 13, 96
    X3 = torch.load('./stress/projection_3.pth')

    conv2_channel = [5, 7, 8, 10, 13, 17, 19, 23, 27, 30]
    conv3_channel = [0, 11, 13, 14, 18, 22, 23, 24, 29, 30, 31, 34, 35, 37, 40, 43, 47, 49, 51, 54, 56, 57, 61]
    W = torch.stack([model.conv2.weight[i] for i in conv2_channel] + [model.conv3.weight[i] for i in conv3_channel],
                    dim=0)
    w1 = torch.flatten(torch.mean(W[:11], dim=0)).reshape(-1, 1)  # 96, 1
    w2 = torch.flatten(torch.mean(W[11:22], dim=0)).reshape(-1, 1)  # 96, 1
    w3 = torch.flatten(torch.mean(W[22:], dim=0)).reshape(-1, 1)
    return torch.mm(X1, w1).data, torch.mm(X2, w2).data, torch.mm(X3, w3).data


if __name__ == '__main__':
    m1, m2, m3 = extract()
    m4 = torch.mean(mfcc('./voice/4.wav'), dim=1)
    m5 = torch.mean(mfcc('./voice/5.wav'), dim=1)
    m6 = torch.mean(mfcc('./voice/6.wav'), dim=1)

    pc = []
    cos = []
    for m in [m1, m2, m3]:
        m = m.squeeze(1)
        pc.append(np.corrcoef(m4.numpy(), m.numpy())[0, 1])
        cos.append(F.cosine_similarity(m4, m, dim=0).item())

    bar_width = 0.35

    categories = ['Group 1', 'Group 2', 'Group 3']
    # 设置x轴位置
    index = np.arange(len(categories))

    # 绘制第一组数据的柱状图
    plt.bar(index, pc, bar_width, label='PC', color='#dae3f3', edgecolor='black')

    # 绘制第二组数据的柱状图，将x轴位置稍微调整
    plt.bar(index + bar_width+0.05, cos, bar_width, label='COS', color='#fbe5d6', edgecolor='black')

    # 设置x轴标签和标题
    # plt.xlabel('Categories')
    plt.ylabel('Similarity Metrics', fontsize=16)
    plt.ylim(0, 1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # 设置x轴刻度位置
    plt.xticks(0.025 + index + bar_width / 2, categories)

    # 添加图例
    plt.legend(fontsize=14)
    plt.savefig('draw/stress.pdf')
    # 显示图形
    plt.show()