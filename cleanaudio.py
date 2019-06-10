import os
#from tqdm import tqdm
#make aprogress bar for you
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
#VAD -trim signals
def envolope(signal,rate,threshold):
    mask=[]
    signal=pd.Series(signal).apply(np.abs)
    signal_mean=signal.rolling(window=int(rate/10),min_periods=1,center=True).mean()
    for mean in signal_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask
            
    
    

df =pd.read_csv("instruments.csv")
df.set_index("fname",inplace=True)


for f in df.index:
    fpath="wavfiles/"+f
    rate,signal=wavfile.read("wavfiles/"+f)
    df.at[f,"length"]=signal.shape[0]/rate
    
classes=list(np.unique(df.label))
class_dist=df.groupby(["label"])["length"].mean()
fig,ax=plt.subplots()
ax.set_title("Class Distribution",y=1.08)
ax.pie(class_dist,labels=class_dist.index,autopct="%1.1f%%",shadow=False,startangle=90)
ax.axis("equal")
plt.show()
df.reset_index(inplace=True)
#features
signals={}
mfccs={}
for c in classes:
    wav_file=df[df.label==c].iloc[0,0]
    signal,rate=librosa.load(fpath, sr=44100, res_type="kaiser_fast")
    mask=envolope(signal,rate,0.0005)
    signal=signal[mask]
    signals[c]=signal
    mfccs[c]=librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13)
    #logmel = librosa.amplitude_to_db(S=mel)#(13,timestamp)

#clean the files by simple VAD
if len(os.listdir('clean'))==0:
    for f in df.fname:
        signal,rate=librosa.load("wavfiles/"+f,sr=1600)
        mask=envolope(signal,rate,0.0005)
        wavfile.write(filename="clean/"+f,rate=rate,data=signal[mask])
