from pathlib import Path
import numpy, scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import librosa, librosa.display
def extract_features_vocal(p):#(24,31)   
    y,sr=librosa.load(p)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return [
        librosa.feature.zero_crossing_rate(y_harmonic),#(1,31)
        librosa.feature.spectral_centroid(y=y, sr=sr),#(1,31)
        librosa.feature.spectral_rolloff(y=y, sr=sr),#(1,31)
        librosa.feature.spectral_contrast(y=y_harmonic,sr=sr,n_bands=6),#(7,31)
        librosa.feature.spectral_flatness(y=y_harmonic),#(1,31)
        librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)#(13,31)
    ]
def extract_features_breath(p):#(3,31)
    y,sr=librosa.load(p)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    mfcc=librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)[12]
    return [
        mfcc,#(1,31)
        librosa.feature.delta(mfcc),#(1,31)
        librosa.feature.delta(librosa.feature.rms(y=y))#(1,31)
    ]

#iterate to load all audio
for p in Path(r'C:\Users\gaoyu\Desktop\feature\records').glob("**/*.wav"):
    audio_features = numpy.array(extract_features_breath(p))
    feature_table = numpy.vstack((audio_features))
    print(feature_table.shape) 
    print(feature_table)
