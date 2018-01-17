"""
train with pretrained model on test
"""
from senet import SeModel
from vgg1d import vgg1d
from trainer import train_model
from data import preprocess_mel, preprocess_wav

BAGGING_NUM=5

print("################################################")
print("Start pretraining models with test data........")
print("################################################")

def train_and_predict(cfg_dict, preprocess_list):
    for p, preprocess_fun in preprocess_list:
        cfg = cfg_dict.copy()
        cfg['preprocess_fun'] = preprocess_fun
        cfg['CODER'] += '_%s' %p
        cfg['bagging_num'] = BAGGING_NUM
        print("training ", cfg['CODER'])
        train_model(**cfg)

se_pretrain = {
    'model_class': SeModel,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 16,
    'epochs': 50,
    'CODER': 'senet_pretrained',
    'pretraining': True,
    'bagging_num': 1
}

print("pretrain senet..........")
#train_and_predict(se_pretrain, [('mel', preprocess_mel)])

vgg1d_pretrain = {
    'model_class': vgg1d,
    'is_1d': True,
    'reshape_size': None,
    'BATCH_SIZE': 32,
    'epochs': 50,
    'CODER': 'vgg1d_pretrained',
    'pretraining': True,
    'bagging_num': 1
}

print("pretrain vgg1d on raw features..........")
#train_and_predict(vgg1d_pretrain, [('raw', preprocess_wav)])


print("################################################")
print("Start fine tuning...")
print("################################################")
se_config = {
    'model_class': SeModel,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 16,
    'epochs': 1,
    'CODER': 'senet_finetune',
    'pretrained': 'model/model_senet_pretrained_mel_0.pth'
}

print("train senet..........")
train_and_predict(se_config, [('mel', preprocess_mel)])

vgg1d_config = {
    'model_class': vgg1d,
    'is_1d': True,
    'reshape_size': None,
    'BATCH_SIZE': 32,
    'epochs': 1,
    'CODER': 'vgg1d_finetune',
    'pretrained': 'model/model_vgg1d_pretrained_raw_0.pth'
}

print("pretrain vgg1d on raw features..........")
train_and_predict(vgg1d_config, [('raw', preprocess_wav)])
