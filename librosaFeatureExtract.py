# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:43:13 2019

@author: gaoyu
"""
import librosa
import librosa.display
import IPython
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

#load video
#audio = librosa.util.example_audio_file() 
audio = r"C:\Users\gaoyu\Desktop\feature\test.wav"
y,sr=librosa.load(audio)

#basic info
print('Audio Sampling Rate: '+str(sr)+' samples/sec')
print('Total Samples: '+str(np.size(y)))
secs=np.size(y)/sr
print('Audio Length: '+str(secs)+' s')
#IPython.display.Audio(audio)

#seperation of harmonic and percussive
y_harmonic, y_percussive = librosa.effects.hpss(y)
'''
plt.figure(figsize=(15, 5))
librosa.display.waveplot(y_harmonic, sr=sr,color="g", alpha=0.25)
librosa.display.waveplot(y_percussive, sr=sr, color='r', alpha=0.5)
plt.title('Harmonic + Percussive')
'''
#(A)timbral texture:
#1)zero crossing
zrate=librosa.feature.zero_crossing_rate(y_harmonic)

zrate_mean=np.mean(zrate)
zrate_std=np.std(zrate)
zrate_skew=scipy.stats.skew(zrate,axis=1)[0]
print('zero-crossing rate::')
print('Mean: '+str(zrate_mean))
print('SD: '+str(zrate_std))
print('Skewness: '+str(zrate_skew))
'''
plt.figure(figsize=(15,5))
plt.semilogy(zrate.T, label='Fraction')
plt.ylabel('Fraction per Frame')
plt.xticks([])
plt.xlim([0, rolloff.shape[-1]])
plt.legend()
'''
#2)spectral centroid
cent = librosa.feature.spectral_centroid(y=y, sr=sr)

cent_mean=np.mean(cent)
cent_std=np.std(cent)
cent_skew=scipy.stats.skew(cent,axis=1)[0]
print('spectral centroid:')
print('Mean: '+str(cent_mean))
print('SD: '+str(cent_std))
print('Skewness: '+str(cent_skew))
'''
plt.figure(figsize=(15,5))
plt.subplot(1, 1, 1)
plt.semilogy(cent.T, label='Spectral centroid')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, cent.shape[-1]])
plt.legend()
'''
#3)spectral flux
#4)spectral rolloff
print('spectral round-off:')
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

cent_mean=np.mean(cent)
cent_std=np.std(cent)
cent_skew=scipy.stats.skew(cent,axis=1)[0]
print('Mean: '+str(cent_mean))
print('SD: '+str(cent_std))
print('Skewness: '+str(cent_skew))
'''
plt.figure(figsize=(15,5))
plt.semilogy(rolloff.T, label='Roll-off frequency')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, rolloff.shape[-1]])
plt.legend()
'''
#5)Mel-frequency cepstral coefficients (MFCC)
mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)


mfccs_mean=np.mean(mfccs,axis=1)
mfccs_std=np.std(mfccs,axis=1)

coeffs=np.arange(0,13)
'''
plt.figure(figsize=(15,5))
plt.title('Mean MFCCs')
sns.barplot(x=coeffs,y=mfccs_mean)

plt.figure(figsize=(15,5))
plt.title('SD MFCCs')
sns.barplot(x=coeffs,y=mfccs_std)
'''
for i in range(0,13):
    print('mfccs_mean_'+str(i)+": "+str(mfccs_mean[i]))
for i in range(0,13):
    print('mfccs_std_'+str(i)+": "+str(mfccs_mean[i]))


'''
plt.figure(figsize=(15, 5))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
'''
#6)Daubechies wavelet coefficients histograms (DWCH)
#7)octave-based spectral contrast (OSC),
#8)spectral contrast
contrast=librosa.feature.spectral_contrast(y=y_harmonic,sr=sr)

contrast_mean=np.mean(contrast,axis=1)
print("constrast mean:")
print(contrast_mean)
contrast_std=np.std(contrast,axis=1)
print("SD contrast:")
print(contrast_std)
'''
plt.figure(figsize=(15,5))
librosa.display.specshow(contrast, x_axis='time')
plt.colorbar()
plt.ylabel('Frequency bands')
plt.title('Spectral contrast')
'''

#(B)rhythmic features:
#1)beat
#tempo(beats/min)
#and an array of frame numbers corresponding to detected beat events.
tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
print('Detected Tempo: '+str(tempo)+ ' beats/min')
'''
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
beat_time_diff=np.ediff1d(beat_times)
beat_nums = np.arange(1, np.size(beat_times))

fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
ax.set_ylabel("Time difference (s)")
ax.set_xlabel("Beats")
g=sns.barplot(beat_nums, beat_time_diff, palette="BuGn_d",ax=ax)
g=g.set(xticklabels=[])
'''
#2)strength

#(C)pitch content features.
#1)pitch histograom
#2)CENS-->smoothed harmonic progression(robust to variation due to timber,articulation)
chroma=librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
#print("CENS:",chroma)
'''
plt.figure(figsize=(15, 5))
librosa.display.specshow(chroma,y_axis='chroma', x_axis='time')
plt.colorbar()
'''
chroma_mean=np.mean(chroma,axis=1)
chroma_std=np.std(chroma,axis=1)

#Generate the chroma Dataframe
chroma_df=pd.DataFrame()
for i in range(0,12):
    chroma_df['chroma_mean_'+str(i)]=chroma_mean[i]
for i in range(0,12):
    chroma_df['chroma_std_'+str(i)]=chroma_mean[i]
chroma_df.loc[0]=np.concatenate((chroma_mean,chroma_std),axis=0)

print(chroma_df)
#plot the summary
'''
octave=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
plt.figure(figsize=(15,5))
plt.title('Mean CENS')
sns.barplot(x=octave,y=chroma_mean)

plt.figure(figsize=(15,5))
plt.title('SD CENS')
sns.barplot(x=octave,y=chroma_std)
'''
