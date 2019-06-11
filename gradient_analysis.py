from __future__ import print_function
from keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2' 
from keras.models import model_from_json
from feature_extractor import *
import numpy as np
import tensorflow as tf

# load model architecture
model_json = "".join(open('./model_architecture').readlines())
model = model_from_json(model_json)
# load model weights
model.load_weights('./model_79_0.00954.h5')
#print(model.summary())
# Parameters for feature extractor
params = [22050, 4096, 128, 1032, 120, 'log_mel']
###############################################################################
# extract feature
feat = feat_ext_fun('../music/1.mp4', params)
# reshape feature
feat = feat.reshape(1,1,feat.shape[0], feat.shape[1])
print(feat.shape)
# predict
preds = model.predict(feat)
pred_class = np.argmax(preds[0])
# 預測分類的輸出向量
pred_output = model.output[:, pred_class]
# 最後一層 convolution layer 輸出的 feature map
# ResNet 的最後一層 convolution layer
last_conv_layer = model.layers[0]
# 求得分類的神經元對於最後一層 convolution layer 的梯度
grads = K.gradients(pred_output, last_conv_layer.output)[0]
print(grads)
# 求得針對每個 feature map 的梯度加總
# pooled_grads = K.sum(grads, axis=(0, 1))
iterate = K.function([model.input], [grads, last_conv_layer.output[0]])
# 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
grads_value, conv_layer_output_value = iterate([feat])
print("\n\n\n\n")
print(grads_value)
np.save("grad",grads_value)
