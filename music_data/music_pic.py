# 存譜圖
music = 
y, sr = librosa.load('./audio_wav/巴赫.wav')

folderPath = 'jpg'
if not os.path.exists(folderPath):
    os.makedirs(folderPath)
    
save_path_wave = '巴赫_wave.jpg'
save_path_mel = '巴赫_mel.jpg'
save_path_zero = '巴赫_zero.jpg'
save_path_centroid = '巴赫_centroid.jpg'
save_path_rolloff = '巴赫_rolloff.jpg'


# # =======
# # no axis
pylab.axis('off') 
# # 刪除空白邊界
pylab.axes([0., 0., 1., 1.], frameon=False) #, xticks=[], ysticks=[]
S = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
pl.savefig(save_path_mel) # , bbox_inches=None, pad_inches=0
pl.close()
# # =======

# # wave
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr) 
plt.title('waveform')
pl.savefig(save_path_wave) # , bbox_inches=None, pad_inches=0
pl.close()

# # zero_crossing
n0 = 9000
n1 = 9100
plt.plot(y[n0:n1])
plt.grid()
# #zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
# #print(sum(zero_crossings))
pl.savefig(save_path_zero) # , bbox_inches=None, pad_inches=0
pl.close()


spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)[0]
spectral_centroids.shape
(775,)

# # Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# # Normalising the spectral centroid for visualisation
def normalize(y, axis=0):
    return sklearn.preprocessing.minmax_scale(y, axis=axis)

# #Plotting the Spectral Centroid along the waveform
librosa.display.waveshow(y, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
pl.savefig(save_path_centroid) # , bbox_inches=None, pad_inches=0
pl.close()


spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr)[0]
librosa.display.waveshow(y, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
pl.savefig(save_path_rolloff) # , bbox_inches=None, pad_inches=0
pl.close()


# 偵測節拍
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
# 節奏頻率（次/分鐘）
print(beat_frames)
# 將 frames 轉為實際時間
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# 節拍的時間點
print(beat_times)