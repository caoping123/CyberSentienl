import librosa
import noisereduce as nr
import soundfile as sf

# 读取音频文件
file_path = './voice/noise/rain.wav'
audio, sr = librosa.load(file_path, sr=None)

# 使用noisereduce库进行降噪
noisy_part = audio  # 你可以选择处理音频的特定部分
reduced_noise = nr.reduce_noise(audio, sr)

# 可选：保存降噪后的音频文件
output_file_path = './voice/noise/rain_reduced.wav'
sf.write(output_file_path, reduced_noise, samplerate=sr)
