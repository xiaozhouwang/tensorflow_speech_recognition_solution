from resnet import ResModel
from senet import SeModel
from vgg2d import vgg2d
from vgg1d import vgg1d, vggmel
from densenet import densenet121
from trainer import train_model
from data import preprocess_mel, preprocess_mfcc, preprocess_wav

list_2d = [('mel', preprocess_mel), ('mfcc', preprocess_mfcc)]
BAGGING_NUM=4

def train_and_predict(cfg_dict, preprocess_list):
    for p, preprocess_fun in preprocess_list:
        cfg = cfg_dict.copy()
        cfg['preprocess_fun'] = preprocess_fun
        cfg['CODER'] += '_%s' %p
        cfg['bagging_num'] = BAGGING_NUM
        print("training ", cfg['CODER'])
        train_model(**cfg)


res_config = {
    'model_class': ResModel,
    'is_1d': False,
    'reshape_size': None,
    'BATCH_SIZE': 32,
    'epochs': 100,
    'CODER': 'resnet'
}

print("train resnet.........")
train_and_predict(res_config, list_2d)

se_config = {
    'model_class': SeModel,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 16,
    'epochs': 100,
    'CODER': 'senet'
}

print("train senet..........")
train_and_predict(se_config, list_2d)

dense_config = {
    'model_class': densenet121,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 16,
    'epochs': 100,
    'CODER': 'dense'
}

print("train densenet.........")
train_and_predict(dense_config, list_2d)

vgg2d_config = {
    'model_class': vgg2d,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 32,
    'epochs': 100,
    'CODER': 'vgg2d'
}

print("train vgg2d...........")
train_and_predict(vgg2d_config, list_2d)

vgg1d_config = {
    'model_class': vgg1d,
    'is_1d': True,
    'reshape_size': None,
    'BATCH_SIZE': 32,
    'epochs': 100,
    'CODER': 'vgg1d'
}

print("train vgg1d on raw features..........")
train_and_predict(vgg1d_config, [('raw', preprocess_wav)])

vggmel_config = {
    'model_class': vggmel,
    'is_1d': True,
    'reshape_size': None,
    'BATCH_SIZE': 64,
    'epochs': 100,
    'CODER': 'vgg1d'
}

print("train vgg1d on mel features..........")
train_and_predict(vggmel_config, [('mel', preprocess_mel)])


