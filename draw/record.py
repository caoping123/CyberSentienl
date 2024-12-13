import librosa
import matplotlib.pyplot as plt
import numpy as np


def spectrum(filepath, savepath=None):
    # 读取语音文件
    audio_path = filepath
    y, sr = librosa.load(audio_path)

    # 计算短时傅里叶变换（STFT）
    D = librosa.stft(y)

    # 计算幅度谱
    magnitude = np.abs(D)

    # 获取频率数组
    frequencies = librosa.fft_frequencies(sr=sr)

    # 选择频率范围在前4000 Hz内的部分
    frequencies = frequencies[frequencies <= 4000]

    # 取相应频率范围内的幅度谱
    magnitude = magnitude[:len(frequencies), :]

    # 取幅度谱的均值作为强度
    intensity = np.mean(magnitude, axis=1)

    # 绘制频率分布图
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, intensity)
    plt.fill_between(frequencies, intensity, color='blue', alpha=0.3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.grid(True)
    if savepath:
        plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    spectrum('../voice/1.wav')
