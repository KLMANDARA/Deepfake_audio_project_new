
import os, librosa, numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def extract_spectrogram(file):
    y, sr = librosa.load(file, sr=16000)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.power_to_db(spec)

def load_data(path):
    X, y = [], []
    for label, folder in enumerate(["real","fake"]):
        folder_path = os.path.join(path, folder)
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            spec = extract_spectrogram(file_path)
            spec = np.resize(spec, (128,128))
            X.append(spec)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data("data")
X = X[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

model.save("model.h5")
