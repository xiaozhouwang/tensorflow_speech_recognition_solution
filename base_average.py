"""
average all base models
"""
import pandas as pd
import numpy as np
from data import get_label_dict

dense_mel = pd.read_csv("pred_scores/dense_mel.csv", index_col=-1)
dense_mfcc = pd.read_csv("pred_scores/dense_mfcc.csv", index_col=-1)

resnet_mel = pd.read_csv("pred_scores/resnet_mel.csv", index_col=-1)
resnet_mfcc = pd.read_csv("pred_scores/resnet_mfcc.csv", index_col=-1)

senet_mel = pd.read_csv("pred_scores/senet_mel.csv", index_col=-1)
senet_mfcc = pd.read_csv("pred_scores/senet_mfcc.csv", index_col=-1)

vgg2d_mel = pd.read_csv("pred_scores/vgg2d_mel.csv", index_col=-1)
vgg2d_mfcc = pd.read_csv("pred_scores/vgg2d_mfcc.csv", index_col=-1)

vgg1d_mel = pd.read_csv("pred_scores/vgg1d_mel.csv", index_col=-1)
vgg1d_raw = pd.read_csv("pred_scores/vgg1d_raw.csv", index_col=-1)


fname = vgg1d_raw.index

"""Weights were determined by using public LB feedback"""
result = (dense_mel.as_matrix() * 0.6 + dense_mfcc.as_matrix() * 0.4) * 0.15 + \
        (resnet_mel.as_matrix() * 0.6 + resnet_mfcc.as_matrix() * 0.4) * 0.15 + \
        (senet_mel.as_matrix() * 0.6 + senet_mfcc.as_matrix() * 0.4) * 0.15 + \
        (vgg2d_mel.as_matrix() * 0.6 + vgg2d_mfcc.as_matrix() * 0.4) * 0.15 + \
        (vgg1d_raw.as_matrix() * 0.75 + vgg1d_mel.as_matrix() * 0.25) * 0.4

label_to_int, int_to_label = get_label_dict()
final_labels = [int_to_label[x] for x in np.argmax(result, 1)]

print(len(final_labels))

labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']
pred_scores = pd.DataFrame(result, columns=labels)
pred_scores['fname'] = fname
pred_scores.to_csv("pred_scores/base_average.csv", index=False)
pd.DataFrame({'fname': fname,
              'label': final_labels}).to_csv("sub/base_average.csv", index=False)
