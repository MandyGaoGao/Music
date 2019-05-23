import librosa
import numpy as np
import os
import sys
import h5py
import multiprocessing
import functools
import h5py
import threading
from pathlib import Path
from numpy import pi, convolve
from scipy.signal.filter_design import bilinear
from scipy.signal import lfilter

#utilities:
def pitch_detect_simple(y, sr):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    shape = pitches.shape
    nb_samples = shape[0]
    nb_windows = shape[1]
    pitches, magnitudes = extract_max(pitches, magnitudes, shape)
    pitches = smooth(pitches, window_len=5)
    return pitches

def harmonic_extract(y, sr):
    h_range = [1]
    S = np.abs(librosa.stft(y))
    fft_freqs = librosa.fft_frequencies(sr=sr)
    S_harm = librosa.interp_harmonics(S, fft_freqs, h_range, axis=0)[0]
    return S_harm

def a_weighting_coeffs_design(sample_rate):
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    numerators = [(2*pi*f4)**2 * (10**(A1000 / 20.0)), 0., 0., 0., 0.];
    denominators = convolve(
        [1., +4*pi * f4, (2*pi * f4)**2],
        [1., +4*pi * f1, (2*pi * f1)**2]
    )
    denominators = convolve(
        convolve(denominators, [1., 2*pi * f3]),
        [1., 2*pi * f2]
    )
    return bilinear(numerators, denominators, sample_rate)

def AweightPower_extract(y, sr):
    b, a = a_weighting_coeffs_design(sr)
    k = lfilter(b, a, y)
    a_weighted_power=librosa.feature.rms(y=k)
    return a_weighted_power


#extract features for each dimension:

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# configure for librosa
class Config(object):
    n_fft = 1024
    hop_length=512
    sr = 22050
    duration = 120
    feature = "mel"

conf = Config()


def extract_max(pitches, magnitudes, shape):
    new_pitches = []
    new_magnitudes = []
    for i in range(0, shape[1]):
        new_pitches.append(np.max(pitches[:,i]))
        new_magnitudes.append(np.max(magnitudes[:,i]))
    return (new_pitches,new_magnitudes)


def smooth(x, window_len=11, window='hanning'):
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]


def pitch_detect(S, sr, fmin, fmax):
    pitches, magnitudes = librosa.core.piptrack(S=S, sr=sr, fmin=fmin, fmax=fmax)
    shape = pitches.shape
    nb_samples = shape[0]
    nb_windows = shape[1]
    pitches, magnitudes = extract_max(pitches, magnitudes, shape)
    pitches = smooth(pitches, window_len=5)
    return pitches


def simple_feature(fpath):
    try:
        x, sr = librosa.load(fpath, sr=44100, res_type="kaiser_fast")
    except:
        print("Fail to decode")
        return None
    x_trimmed, _ = librosa.effects.trim(x, top_db=30)

    S = np.abs(librosa.stft(x_trimmed, n_fft=1024, hop_length=512))



def hand_feature(fpath):
    try:
        x, sr = librosa.load(fpath, sr=conf.sr, res_type="kaiser_fast")
    except:
        print("Fail to decode")
        return None
    x_trimmed, _ = librosa.effects.trim(x, top_db=30)

    D = librosa.stft(x_trimmed, n_fft=1024, hop_length=512)
    S = np.abs(D)
    S_harm, S_perc = librosa.decompose.hpss(D)
    S_harm = np.abs(S_harm)
    S_perc = np.abs(S_perc)
    # onset-env
    onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S_perc))
    onset_env = np.expand_dims(onset_env, axis=1)

    # energy(root-mse)
    rmse = librosa.feature.rmse(S=S).T
    #rmse = np.expand_dims(rmse, axis=1)

    # zcr 
    zcr = librosa.feature.zero_crossing_rate(x_trimmed, frame_length=1024, hop_length=512).T
    #zcr = np.expand_dims(zcr, axis=1)

    # timbre: to compute

    # chroma: chords
    chroma = librosa.feature.chroma_stft(S=S_harm, sr=sr)
    chroma = chroma.T

    feat = np.hstack([onset_env, rmse, zcr, chroma])

    feat_mean = np.mean(feat, axis=0)
    feat_std = np.std(feat, axis=0)
    feat = (feat - feat_mean)/(feat_std+1e-10)

    print(fpath, feat.shape)

    return feat


def preprocess(fpath):
    try:
        x, sr = librosa.load(fpath, sr=conf.sr, res_type="kaiser_fast")
    except:
        print("Fail to decode")
        return None
    x_trimmed, _ = librosa.effects.trim(x, top_db=30)

    S = np.abs(librosa.stft(x_trimmed, n_fft=1024, hop_length=512))
    mel = librosa.feature.melspectrogram(S=S)
    logmel = librosa.amplitude_to_db(S=mel)
    freqs = librosa.core.fft_frequencies(sr)
    harms = [1, 2, 3, 4]
    weights = [1.0, 0.5, 0.33, 0.25]

    feat1 = logmel
    #feat1 = librosa.feature.mfcc(S=logmel)
    feat1 = feat1.T

    '''
    feat2 = librosa.feature.spectral_centroid(S=S)
    feat2 = feat2.T
    feat3 = librosa.feature.spectral_bandwidth(S=S)
    feat3 = feat3.T
    feat4 = librosa.feature.spectral_flatness(S=S)
    feat4 = feat4.T
    feat5 = librosa.feature.spectral_contrast(S=S)
    feat5 = feat5.T
    feat6 = librosa.feature.zero_crossing_rate(x_trimmed, frame_length=1024, hop_length=512)
    feat6 = feat6.T
    '''

    feat7 = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S))
    feat7 = np.expand_dims(feat7, axis=1)
    feat8 = np.expand_dims(pitch_detect(S, sr, 300, 3000), axis=1)

    """
    print("mfcc shape ", feat1.shape)
    print("spec centriod shape ", feat2.shape)
    print("spec bandwidth shape ", feat3.shape)
    print("spec flatness shape", feat4.shape)
    print("spec contrast shape", feat5.shape)
    print("spec zcr shape", feat6.shape)
    print("onset_strength shape", feat7.shape)
    """

    #feat = np.hstack([feat1,feat2,feat3, feat4, feat5,feat6,feat7, feat8])
    feat = np.hstack([feat1, feat7, feat8])

    feat_mean = np.mean(feat, axis=0)
    feat_std = np.std(feat, axis=0)
    feat = (feat - feat_mean)/(feat_std+1e-10)
    #feat = (feat - np.expand_dims(feat_mean,0))/(feat_std[None,:] + 1e-10)
    print(fpath, feat.shape)

    return feat

def vocal_feature(fpath):
    try:
        x, sr = librosa.load(fpath, sr=conf.sr, res_type="kaiser_fast")
    except:
        print("Fail to decode")
        return None
    x_trimmed, _ = librosa.effects.trim(x, top_db=30)

    # timbre: to compute
    zero=librosa.feature.zero_crossing_rate(y=x_trimmed)#(1,31)
    centroid=librosa.feature.spectral_centroid(y=x_trimmed, sr=sr)#(1,31)
    rolloff=librosa.feature.spectral_rolloff(y=x_trimmed, sr=sr)#(1,31)
    contrast=librosa.feature.spectral_contrast(y=x_trimmed,sr=sr,n_bands=6)#(7,31)
    flatness=librosa.feature.spectral_flatness(y=x_trimmed)#(1,31)
    mfccs=librosa.feature.mfcc(y=x_trimmed, sr=sr, n_mfcc=13)#(13,31)
    audio_features = np.array([zero, centroid, rolloff,contrast,flatness,mfccs])   #(1028,31)
    feat= np.vstack(audio_features)
    feat_mean=np.mean(feat, axis=0)
    feat_std = np.std(feat, axis=0)
    feat = (feat - feat_mean)/(feat_std+1e-10)

    print(fpath, feat.shape)
    return feat

def breath_feature(fpath):
    try:
        x, sr = librosa.load(fpath, sr=conf.sr, res_type="kaiser_fast")
    except:
        print("Fail to decode")
        return None
    x_trimmed, _ = librosa.effects.trim(x, top_db=30)
    
    mfcc=librosa.feature.mfcc(y=x_trimmed, sr=sr, n_mfcc=13)[12] 
    delta_mfcc=librosa.feature.delta(mfcc)#(1,31)
    delta_energy=librosa.feature.delta(librosa.feature.rms(y=x_trimmed))#(1,31)
    audio_features = np.array([mfcc,delta_mfcc, delta_energy])   #(1028,31)
    feat= np.vstack(audio_features)

    feat_mean = np.mean(feat, axis=0)
    feat_std = np.std(feat, axis=0)
    feat = (feat - feat_mean)/(feat_std+1e-10)

    print(fpath, feat.shape)
    return feat

def pitch_feature(fpath):
    try:
        x, sr = librosa.load(fpath, sr=conf.sr, res_type="kaiser_fast")
    except:
        print("Fail to decode")
        return None
    x_trimmed, _ = librosa.effects.trim(x, top_db=30)
    pitches=pitch_detect_simple(y=x_trimmed, sr=sr)#(1,31)   
    audio_features = np.array([pitches])   #(1028,31)
    feat= np.vstack(audio_features)
    feat = np.hstack([pitches])
    feat_mean = np.mean(feat, axis=0)
    feat_std = np.std(feat, axis=0)
    feat = (feat - feat_mean)/(feat_std+1e-10)

    print(fpath, feat.shape)
    return feat

def rhyme_feature(fpath):
    try:
        x, sr = librosa.load(fpath, sr=conf.sr, res_type="kaiser_fast")
    except:
        print("Fail to decode")
        return None
    x_trimmed, _ = librosa.effects.trim(x, top_db=30)
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=x_trimmed, sr=sr, hop_length=hop_length)
    tempo=librosa.feature.tempogram(onset_envelope=oenv, sr=sr,hop_length=hop_length)#(384,31)
    chroma=librosa.feature.chroma_cens(y=x_trimmed, sr=sr)
    audio_features = np.array([tempo,chroma])   
    feat= np.vstack(audio_features) 
    feat_mean = np.mean(feat, axis=0)
    feat_std = np.std(feat, axis=0)
    feat = (feat - feat_mean)/(feat_std+1e-10)

    print(fpath, feat.shape)
    return feat

def emotion_feature(fpath):
    try:
        x, sr = librosa.load(fpath, sr=conf.sr, res_type="kaiser_fast")
    except:
        print("Fail to decode")
        return None
    x_trimmed, _ = librosa.effects.trim(x, top_db=30)
    aweightPower=AweightPower_extract(x_trimmed, sr)#(1,31)
   
    power=librosa.feature.rms(y=x_trimmed)#(1,31)intensity

    harmonics=librosa.feature.zero_crossing_rate(y=x_trimmed)#(1,31)
  
    audio_features = np.array([aweightPower,power,harmonics]) #(1028,31)
    feat= np.vstack(audio_features)
    feat_mean = np.mean(feat, axis=0)
    feat_std = np.std(feat, axis=0)
    feat = (feat - feat_mean)/(feat_std+1e-10)

    print(fpath, feat.shape)
    return feat



def get_feature_extract_func(k):
    if k=="melmix":
        return preprocess
    elif k=="hand":
        return hand_feature

    elif k=="vocal":
        return vocal_feature
    elif k=="breath":
        return breath_feature
    elif k=="pitch":
        return pitch_feature
    elif k=="rhyme":
        return rhyme_feature
    elif k=="emotion":
        return emotion_feature
    

if __name__ == "__main__":

    tester = r"C:\Users\gaoyu\Desktop\feature\test.wav"
    #feat = hand_feature(tester)#(26,15)

    #feat = vocal_feature(tester)#(24,26)
    #feat = breath_feature(tester)#(3,26)
    #feat = pitch_feature(tester)#(1,26)
    #feat = emotion_feature(tester)#(3,26)                                                     188,1         Bot

                                                                                                                     128,2         62%
