import librosa
# 畫音訊陣列
import matplotlib.pyplot as plt
# 不要產生譜圖
# matplotlib.use('Agg')
import librosa.display
import pylab as pl
import sklearn
import numpy as np
import os


'''過零率'''

# 讀取音檔.
x, sr = librosa.load('./audio_wav/lala.wav')
#x, sr = librosa.load('./party/BLACKPINK - FOREVER YOUNG (Color Coded Lyrics EngRomHan가사 ).wav')

# 畫頻譜圖
plt.figure(figsize=(14, 5))
# plt.figure()
librosa.display.waveshow(x, sr=sr)  # x:時間, sr:取樣率
plt.title('lala waveform')
# plt.show()

n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))

y, sr = librosa.load('./audio_wav/lala.wav')
librosa.feature.zero_crossing_rate(y)


'''光譜質心'''
x, sr = librosa.load('./audio_wav/lala.wav')
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
(775,)

# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# Plotting the Spectral Centroid along the waveform
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')


'''光譜衰減'''
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')

'''梅爾倒頻譜係數'''
x, fs = librosa.load('./audio_wav/lala.wav')
librosa.display.waveshow(x, sr=sr)
# 計算超過 97 帪的 20 個 MFCC
mfccs = librosa.feature.mfcc(x, sr=fs)
print(mfccs.shape)
(20, 97)
# Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

# 特徵縮放，使每個單位皆有零均值和單位方差
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

'''色度頻率'''
# Loadign the file
x, sr = librosa.load('./audio_wav/lala.wav')
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time',
                         y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
