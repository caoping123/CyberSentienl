import numpy
import torch
from model import M5
from mfcc import mfcc
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import librosa


def extract():
    model = M5()
    model.load_state_dict(torch.load('wm_model/model_43'))
    X = torch.load('projection.pth')

    conv2_channel = [5, 7, 8, 10, 13, 17, 19, 23, 27, 30]
    conv3_channel = [0, 11, 13, 14, 18, 22, 23, 24, 29, 30, 31, 34, 35, 37, 40, 43, 47, 49, 51, 54, 56, 57, 61]
    W = torch.stack([model.conv2.weight[i] for i in conv2_channel] + [model.conv3.weight[i] for i in conv3_channel],
                    dim=0)
    w = torch.flatten(torch.mean(W, dim=0)).reshape(-1, 1).data  # 96, 1
    return torch.mm(X, w)


def seg_wav(filepath):
    m = mfcc(filepath)
    ret =[]
    for i in range(int(m.shape[1] / 100)):
        seg = m[:, i * 100: i * 100 + 100]
        seg = torch.mean(seg, dim=1)
        ret.append(seg)
    return ret




def noise(m):
    mfcc = seg_wav('./voice/noise.wav')
    pc = []
    cos = []
    for seg in mfcc:
        pc.append(numpy.corrcoef(seg.numpy(), m.numpy())[0, 1])
        cos.append(F.cosine_similarity(seg, m, dim=0).item())
    print(pc)
    print(cos)


if __name__ == '__main__':

    m = extract().squeeze(1)
    noise(m)
    exit(0)
    total_user = 10
    mfccs = [seg_wav('./voice/' + str(i) + '.wav') for i in range(1, total_user + 1)]  # [用户[段[特征]]]

    # 每段
    for user in mfccs:
        user_distinct = []
        for seg in user:
            pc = numpy.corrcoef(seg.numpy(), m.numpy())[0, 1]
            cos = F.cosine_similarity(seg, m, dim=0).item()
            euclid = torch.norm(seg - m).item()
            user_distinct.append([pc, cos, euclid])
        csv_file = './seg_dist.csv'
        with open(csv_file, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(user_distinct)

    # 均值
    for user in mfccs:
        pc_distinct = []
        cos_distinct = []
        euclid_distinct = []

        for seg in user:
            pc = numpy.corrcoef(seg.numpy(), m.numpy())[0, 1]
            cos = F.cosine_similarity(seg, m, dim=0).item()
            euclid = torch.norm(seg - m).item()
            pc_distinct.append(pc)
            cos_distinct.append(cos)
            euclid_distinct.append(euclid)

        csv_file = './user_dist.csv'
        with open(csv_file, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([[sum(pc_distinct) / len(pc_distinct), sum(cos_distinct) / len(cos_distinct),
                               sum(euclid_distinct) / len(euclid_distinct)]])

    # for v in voice:
    #     cos_sim = []
    #     for seg in v:
    #         cos_sim.append(F.cosine_similarity(seg, m, dim=0).item())
    #     print(sum(cos_sim) / len(cos_sim))
    # for seg in m1:
    #     cos_sim.append(F.cosine_similarity(seg, m, dim=0).item())
    # for seg in m0:
    #     cos_sim.append(F.cosine_similarity(seg, m, dim=0).item())
    # for seg in m2:
    #     cos_sim.append(F.cosine_similarity(seg, m, dim=0).item())
    #
    # plt.bar(range(len(cos_sim)), cos_sim, color='skyblue')
    # plt.title("cos")
    # plt.show()

    # print()
    # #皮尔逊系数
    # for v in voice:
    #     pc = []
    #     for seg in v:
    #         pc.append(numpy.corrcoef(seg.numpy(), m.numpy())[0, 1])
    #     print(sum(pc) / len(pc))

    # plt.bar(range(len(pc)), pc, color='skyblue')
    # plt.title("pc")
    # plt.show()

    # print()
    # # 欧几里得距离
    # ed = []
    # for v in voice:
    #     for seg in v:
    #         ed.append(torch.norm(seg - m).item())

    # plt.bar(range(len(ed)), ed, color='skyblue')
    # plt.title("ed")
    # plt.show()
