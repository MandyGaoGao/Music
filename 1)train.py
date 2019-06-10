import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Config
import librosa


def check_data():
    if os.path.isfile(config.p_path):
        print("Loading existing data for {} model".format(config.mode))
        with open(config.p.path,"rb") as handle:
            tmp=pickle.load(handle)
            return tmp
    else:
        return None
#process data for training
def build_rand_feat():
    tmp=check_data()
    if tmp:
        return tmp.data[0],tmp.data[1]
    X=[]
    y=[]
    _min,_max=float('inf'),-float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class=np.random.choice(class_dist.index,p=prob_dist)
        file=np.random.choice(df[df.label==rand_class].index)
        rate,wav=wavfile.read("clean/"+file)   
        label=df.at[file,'label']
        rand_index=np.random.randint(0,wav.shape[0]-config.step)
        sample=wav[rand_index:rand_index+config.step]
        #TODO:
        X_sample=librosa.feature.mfcc(y=sample, sr=rate, n_mfcc=13)
        
        _min=min(np.amin(X_sample),_min)
        _max=max(np.amix(X_sample),_max)
        X.append(X_sample)#.T
        y.append(classes.index(label))
    config.min=_min
    config.max=_max    
    X,y=np.array(X),np.array(y)
    X=(X-_min)/(_max-_min)
    if config.mode=="conv":
        X=X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    elif config.mode=="time":
        X=X.reshape(X.shape[0],X.shape[1],X.shape[2])
    #TODO:
    y=to_categorical(y,num_classes=2)
    config.data(X,y)
    
    with open(config.p_path,"wb") as handle:
        pickle.dump(config,handle,protocal=2)
    return X,y

def get_recurrent_model():
    #shape of RNN (n,time,feat)
    model=Sequential()
    model.add(LSTM(128,return_sequences=True,input_shape=input_shape))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64,activtion="relu")))
    model.add(TimeDistributed(Dense(32,activtion="relu")))
    model.add(TimeDistributed(Dense(16,activtion="relu")))
    model.add(TimeDistributed(Dense(8,activtion="relu")))
    model.add(Flatten())
    model.add(Dense(10,activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])
    return model
        
#build model
def get_conv_model():
    model=Sequential()
    model.add(Conv2D(16,(3,3),activation="relu",strides=(1,1),padding="same"))
    model.add(Conv2D(16,(3,3),activation="relu",strides=(1,1),padding="same"))
    model.add(Conv2D(16,(3,3),activation="relu",strides=(1,1),padding="same"))
    model.add(Conv2D(16,(3,3),activation="relu",strides=(1,1),padding="same"))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(10,activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])
    return model


#build two directoies one "model" "pickle" (model data)
config=Config(mode="time")

#read training set+label csv
df = pd.read_csv('instruments.csv')
df.set_index('fpath', inplace=True)#set filepath as index

for f in df.index:#
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
#create a new column of length based on signal length
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show() 
#prep for gen data_feature
n_samples=2*int(df["length"].sum()/0.1)
prob_dist=class_dist/class_dist.sum()


if config.mode=="conv":
    X,y=build_rand_feat()
    y_flat=np.argmax(y,axis=1)
    input_shape=(X.shape[1],X.shape[2],1)
    model=get_conv_model()
    
elif config.mode=="time":
    X,y=build_rand_feat()
    y_flat=np.argmax(y,axis=1)
    #X samples  (number of samples,timestamp,feature dimension,label,1)
    input_shape=(X.shape[1],X.shape[2])
    model=get_recurrent_model()


class_weight=compute_class_weight("balanced",np.unique(y),y_flat)
checkpoint=ModelCheckpoint(config.model_path,monitor="val_acc",verbose=1,mode="max",
                           save_best_only=True,save_weights_only=False,periods=1)
model.fit(X,y,epoch=10,batch_size=32,shuffle=True,
          validation_split=0.1,callbacks=[checkpoint])
model.save(config.model_path)
