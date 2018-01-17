"""
average all base models
"""
import pandas as pd
import numpy as np
from data import get_label_dict

resnet_semi_mel = pd.read_csv("pred_scores/resnet_semi_mel.csv", index_col=-1)
resnet_semi_mfcc = pd.read_csv("pred_scores/resnet_semi_mfcc.csv", index_col=-1)
senet_semi_mel = pd.read_csv("pred_scores/senet_semi_mel.csv", index_col=-1)
senet_semi_mfcc = pd.read_csv("pred_scores/senet_semi_mfcc.csv", index_col=-1)
vgg2d_semi_mel = pd.read_csv("pred_scores/vgg2d_semi_mel.csv", index_col=-1)
vgg2d_semi_mfcc = pd.read_csv("pred_scores/vgg2d_semi_mfcc.csv", index_col=-1)
vgg1d_semi_raw = pd.read_csv("pred_scores/vgg1d_semi_raw.csv", index_col=-1)

vgg1d_finetune = pd.read_csv("pred_scores/vgg1d_finetune_raw.csv", index_col=-1)
senet_finetune = pd.read_csv("pred_scores/senet_finetune_mel.csv", index_col=-1)

base_model = pd.read_csv("pred_scores/base_average.csv", index_col=-1)

fname = vgg1d_semi_raw.index

"""Weights were determined by using public LB feedback"""
result = (resnet_semi_mel.as_matrix() * 0.5 + resnet_semi_mfcc.as_matrix() * 0.5) * 0.1 + \
        (senet_semi_mel.as_matrix() * 0.4 + senet_semi_mfcc.as_matrix() * 0.6) * 0.2 + \
        (vgg2d_semi_mel.as_matrix() * 0.6 + vgg2d_semi_mfcc.as_matrix() * 0.4) * 0.15 + \
         vgg1d_semi_raw.as_matrix() * 0.25 + base_model.as_matrix() * 0.1 + \
        (vgg1d_finetune.as_matrix() * 0.5 + senet_finetune.as_matrix() * 0.5) * 0.2


label_to_int, int_to_label = get_label_dict()
final_labels = [int_to_label[x] for x in np.argmax(result, 1)]

print(len(final_labels))

pd.DataFrame({'fname': fname,
              'label': final_labels}).to_csv("sub/final_average.csv", index=False)
