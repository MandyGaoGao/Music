
import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from keras_models import load_model
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

def build_predictions(audio_dir):
    y_true=[]
    y_pred=[]
    fn_prob={}
    print("extraction features")
    for fn in tqdm(os.listdir(audio_dir)):
        rate,wav=wavfile.read(os.path.join(audio_dir,fn))
        label=fn2class[fn]
        c=classes.index(label)
        y_prob=[]
        for i in range(0,wav.shape[0]-config.stop,config.step):
            sample=wav[i:i+config.step]
            x=librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13).T
            x=(x-config.min)/(config.max-config.min)
            if config.mode=="conv":
                x=x.reshape(1,x.shape[0],x.shape[1],1)
            elif config.mode="time":
                x=np.expend_dims(x,axis=0)
            y_hat=model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
        fn_prob[fn]=np.mean(y_prob,axis=0).flatten()
    return y_true,y_pred,fn_prob

df=pd.read_csv("instruments.csv")
classes=list(np.unique(df.label))
fn2class=dict(zip(df.fname,df,label))
p_path=os.path.join("pickles","conv.p")
with open(p_path,"rb") as handle:
    config=pickle.load(handle)
model=load_model(config.model_path)
y_true,y_pre,fn_prob=build_predictions("clean")
    
acc_score=accurancy_score(y_true=y_true,y_pred=y_pred)
y_prob=[]
for i,row in df.iterrows():
    y_prob=fn_prob[row.fname]
    y_probs.append(y_prob)
    for c,p in zip(classes,y_prob):
        fd.at[i,c]=p
p_pred=[classes[np.argmax(y) for y in y_probs]]
df["y_pred"]=y_pred
df.to_csv("predict_result",index=False)
