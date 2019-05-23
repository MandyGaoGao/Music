
import librosa
import numpy as np
import os
import sys
import h5py
import multiprocessing
import functools
import h5py
import threading

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


def get_feature_extract_func(k):
    if k=="melmix":
        return preprocess

    elif k=="hand":
        return hand_feature

   # elif k ='vocal':
        #return vocal_feature

if __name__ == "__main__":

    tester = "./dataset/tester/tester.mp3"

    feat = hand_feature(tester)





-- INSERT --                                                                                                                                188,1         Bot
