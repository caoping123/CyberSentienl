import librosa
import numpy
import torch
import torch.nn.functional as F


# 读取音频文件
def mfcc(filepath):
    y, sr = librosa.load(filepath)
    # 提取MFCC特征
    return torch.tensor(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)[2:])


if __name__ == '__main__':
    print(mfcc('voice/2.wav')[:, :20])
