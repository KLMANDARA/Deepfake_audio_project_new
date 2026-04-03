
import librosa, numpy as np
from tensorflow.keras.models import load_model

def extract(file):
    y, sr = librosa.load(file, sr=16000)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec = librosa.power_to_db(spec)
    spec = np.resize(spec,(128,128))
    return spec.reshape(1,128,128,1)

model = load_model("model.h5")

file = input("Enter file: ")
pred = model.predict(extract(file))

print("Fake" if pred>0.5 else "Real")
