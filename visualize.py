
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = input("Enter audio file: ")
y, sr = librosa.load(file, sr=16000)

plt.figure()
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.show()

S = librosa.feature.melspectrogram(y=y, sr=sr)
S_db = librosa.power_to_db(S, ref=np.max)

plt.figure()
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
plt.title("Spectrogram")
plt.colorbar()
plt.show()

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
plt.figure()
librosa.display.specshow(mfcc, x_axis='time')
plt.title("MFCC")
plt.colorbar()
plt.show()
