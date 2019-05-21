# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:14:47 2019

@author: gaoyu
"""
#feature with respect to Time
#output:array: N*T
from pathlib import Path
import numpy, scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import librosa, librosa.display
#iterate to load all audio
audio_signals = [
    librosa.load(p)[0] for p in Path(r'C:\Users\gaoyu\Desktop\feature\records').glob("**/*.wav")
]
#sample size
print(len(audio_signals))
#display signals
plt.figure(figsize=(15, 6))
for i, x in enumerate(audio_signals):
    plt.subplot(2, 5, i+1)
    librosa.display.waveplot(x[:10000])
    plt.ylim(-1, 1)
#define feature vector
def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0, 0],
        librosa.feature.spectral_centroid(signal)[0, 0],
    ]
#aggregate feature vector to collection
audio_features = numpy.array([extract_features(x) for x in audio_signals])
#feature scaling
feature_table = numpy.vstack((audio_features))
print(feature_table.shape)
print(feature_table)
#scale: -1 to 1
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
training_features = scaler.fit_transform(feature_table)
print(training_features)
