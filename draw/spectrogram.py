import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft


def plot_spectrum(filepath, f_ratio=0.2, savepath=None):
    signal, sr = librosa.load(filepath)
    x = np.fft.fft(signal)
    x = np.absolute(x)
    x = (x - min(x)) / (max(x) - min(x))
    plt.figure(figsize=(5, 5))
    f = np.linspace(0, sr, len(x))
    f_bins = int(len(x) * f_ratio)

    print(np.trapz(x[:f_bins], f[:f_bins], 0.001))
    plt.plot(f[:f_bins], x[:f_bins])
    if savepath:
        plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    audio_path1 = "../voice/cao.wav"
    audio_path2 = "../voice/cao_record.wav"
    plot_spectrum(audio_path1)
    plot_spectrum(audio_path2)

