import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def pre_emphasis(signal, pre_emphasis_factor=0.97):
    """对音频信号进行预加重处理."""
    return np.append(signal[0], signal[1:] - pre_emphasis_factor * signal[:-1])

# 读取音频文件
file_path = '../voice/1.wav'  # 替换为你的音频文件路径
signal, sr = librosa.load(file_path, sr=None)

# 进行预加重
pre_emphasized_signal = pre_emphasis(signal)
# print(pre_emphasized_signal.shape)
#
# # 绘制预加重前后的波形
# plt.figure(figsize=(14, 8))

# # 绘制原始信号波形
# plt.subplot(2, 1, 1)
# librosa.display.waveshow(signal, sr=sr)
# plt.title('Original Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')

# 绘制预加重后的信号波形
#plt.subplot(2, 1, 2)
# librosa.display.waveshow(pre_emphasized_signal, sr=sr)
# plt.title('Pre-emphasized Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# #plt.xlim(20, 20.1)
# plt.grid(True)
# #plt.tight_layout()
# plt.show()
time_limit = 0.1  # 100毫秒
sample_limit = int(time_limit * sr)

# 对预加重后的信号进行傅里叶变换
fft_spectrum = np.fft.fft(pre_emphasized_signal[20:20+sample_limit])
freq = np.fft.fftfreq(sample_limit, 1/sr)

# 仅保留正频率部分
positive_freqs = freq[:sample_limit // 2]
magnitude_spectrum = np.abs(fft_spectrum[:sample_limit // 2])
print(magnitude_spectrum.shape)

n_mels = 26
mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=sample_limit-1, n_mels=n_mels, fmin=0, fmax=sr/2)

# 应用梅尔滤波器组
mel_spectrum = np.dot(mel_filter_bank, magnitude_spectrum)
print(mel_spectrum.shape)

# 绘制傅里叶变换后的频谱图
plt.figure(figsize=(14, 6))

# 绘制梅尔滤波器组
# plt.subplot(2, 1, 1)
# librosa.display.specshow(librosa.power_to_db(mel_filter_bank, ref=np.max),
#                          sr=sr, x_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel Filter Bank')

# 绘制通过梅尔滤波器组后的频谱图
# plt.subplot(2, 1, 2)
#plt.plot(np.linspace(0, sr / 2, n_mels), mel_spectrum)
print(mel_spectrum)
plt.bar(x = range(26), height=mel_spectrum)
plt.title('Mel Spectrum of Pre-emphasized Signal)')
#plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# plt.tight_layout()
plt.show()