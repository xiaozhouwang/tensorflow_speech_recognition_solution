"""
model trainer
"""
from torch.autograd import Variable
from data import get_label_dict, get_wav_list, SpeechDataset, get_semi_list
from pretrain_data import PreDataset
from torch.utils.data import DataLoader
import torch
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice


def train_model(model_class, preprocess_fun, is_1d, reshape_size, BATCH_SIZE, epochs, CODER, preprocess_param={},
                bagging_num=1, semi_train_path=None, pretrained=None, pretraining=False, MGPU=False):
    """
    :param model_class: model class. e.g. vgg, resnet, senet
    :param preprocess_fun: preprocess function. e.g. mel, mfcc, raw wave
    :param is_1d: boolean. True for conv1d models and false for conv2d
    :param reshape_size: int. only for conv2d, reshape the image size
    :param BATCH_SIZE: batch size.
    :param epochs: number of epochs
    :param CODER: string for saving and loading model/files
    :param preprocess_param: parameters for preprocessing function
    :param bagging_num: number of training per model, aka bagging models
    :param semi_train_path: path to semi supervised learning file.
    :param pretrained: path to pretrained model
    :param pretraining: boolean. if this is pretraining
    :param MGPU: whether using multiple gpus
    """

    def get_model(model=model_class, m=MGPU, pretrained=pretrained):
        mdl = torch.nn.DataParallel(model()) if m else model()
        if not pretrained:
            return mdl
        else:
            print("load pretrained model here...")
            mdl.load_state_dict(torch.load(pretrained))
            if 'vgg' in pretrained:
                fixed_layers = list(mdl.features)
                for l in fixed_layers:
                    for p in l.parameters():
                        p.requires_grad = False
            return mdl

    label_to_int, int_to_label = get_label_dict()
    for b in range(bagging_num):
        print("training model # ", b)

        loss_fn = torch.nn.CrossEntropyLoss()

        speechmodel = get_model()
        speechmodel = speechmodel.cuda()

        total_correct = 0
        num_labels = 0
        start_time = time()

        for e in range(epochs):
            print("training epoch ", e)
            learning_rate = 0.01 if e < 10 else 0.001
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speechmodel.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.00001)
            speechmodel.train()
            if semi_train_path:
                train_list = get_semi_list(words=label_to_int.keys(), sub_path=semi_train_path,
                                           test_ratio=choice([0.2, 0.25, 0.3, 0.35]))
                print("semi training list length: ", len(train_list))
            else:
                train_list, _ = get_wav_list(words=label_to_int.keys())

            if pretraining:
                traindataset = PreDataset(label_words_dict=label_to_int,
                                          add_noise=True, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,
                                          resize_shape=reshape_size, is_1d=is_1d)
            else:
                traindataset = SpeechDataset(mode='train', label_words_dict=label_to_int, wav_list=train_list,
                                             add_noise=True, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,
                                             resize_shape=reshape_size, is_1d=is_1d)
            trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True)
            for batch_idx, batch_data in enumerate(trainloader):
                spec = batch_data['spec']
                label = batch_data['label']
                spec, label = Variable(spec.cuda()), Variable(label.cuda())
                y_pred = speechmodel(spec)
                _, pred_labels = torch.max(y_pred.data, 1)
                correct = (pred_labels == label.data).sum()
                loss = loss_fn(y_pred, label)

                total_correct += correct
                num_labels += len(label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("training loss:", 100. * total_correct / num_labels, time()-start_time)

        # save model
        create_directory("model")
        torch.save(speechmodel.state_dict(), "model/model_%s_%s.pth" % (CODER, b))

    if not pretraining:
        print("doing prediction...")
        softmax = Softmax()

        trained_models = ["model/model_%s_%s.pth" % (CODER, b) for b in range(bagging_num)]

        # prediction
        _, test_list = get_wav_list(words=label_to_int.keys())
        testdataset = SpeechDataset(mode='test', label_words_dict=label_to_int, wav_list=test_list,
                                    add_noise=False, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,
                                    resize_shape=reshape_size, is_1d=is_1d)
        testloader = DataLoader(testdataset, BATCH_SIZE, shuffle=False)

        for e, m in enumerate(trained_models):
            print("predicting ", m)
            speechmodel = get_model(m=MGPU)
            speechmodel.load_state_dict(torch.load(m))
            speechmodel = speechmodel.cuda()
            speechmodel.eval()

            test_fnames, test_labels = [], []
            pred_scores = []
            # do prediction and make a submission file
            for batch_idx, batch_data in enumerate(testloader):
                spec = Variable(batch_data['spec'].cuda())
                fname = batch_data['id']
                y_pred = softmax(speechmodel(spec))
                pred_scores.append(y_pred.data.cpu().numpy())
                test_fnames += fname

            if e == 0:
                final_pred = np.vstack(pred_scores)
                final_test_fnames = test_fnames
            else:
                final_pred += np.vstack(pred_scores)
                assert final_test_fnames == test_fnames

        final_pred /= len(trained_models)
        final_labels = [int_to_label[x] for x in np.argmax(final_pred, 1)]

        test_fnames = [x.split("/")[-1] for x in final_test_fnames]

        labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']
        pred_scores = pd.DataFrame(np.vstack(final_pred), columns=labels)
        pred_scores['fname'] = test_fnames

        create_directory("pred_scores")
        pred_scores.to_csv("pred_scores/%s.csv" % CODER, index=False)

        create_directory("sub")
        pd.DataFrame({'fname': test_fnames,
                      'label': final_labels}).to_csv("sub/%s.csv" % CODER, index=False)


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
