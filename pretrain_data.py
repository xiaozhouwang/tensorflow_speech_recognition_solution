"""
test dataset for initializing weights
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
from glob import glob
import random
from skimage.transform import resize
from random import choice, sample
import pandas as pd

SR = 16000


class PreDataset(Dataset):
    """
    1. add background noise
    2. generate silent data
    3. cache some parts to speed up iterating
    """

    def __init__(self, label_words_dict, add_noise, preprocess_fun, preprocess_param = {}, sr=SR, resize_shape=128, is_1d=False):
        self.add_noise = add_noise
        self.sr = sr
        self.label_words_dict = label_words_dict
        self.preprocess_fun = preprocess_fun
        self.preprocess_param = preprocess_param

        # read all background noise here
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("../input/train/audio/_background_noise_/*.wav")]

        self.resize_shape = resize_shape
        self.is_1d = is_1d
        pre_list = pd.read_csv("sub/base_average.csv")
        self.semi_dict = dict(zip(pre_list['fname'], pre_list['label']))
        self.wav_list = ['../input/test/audio/' + x for x in self.semi_dict]
        self.wav_list = sample(self.wav_list, len(self.wav_list))

    def get_one_noise(self):
        """generates one single noise clip"""
        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]
        # only takes out 16000
        start_idx = random.randint(0, len(selected_noise)-1-self.sr)
        return selected_noise[start_idx:(start_idx + self.sr)]

    def get_mix_noises(self, num_noise=1, max_ratio=0.1):
        result = np.zeros(self.sr)
        for _ in range(num_noise):
            result += random.random() * max_ratio * self.get_one_noise()
        return result / num_noise if num_noise > 0 else result

    def get_one_word_wav(self, idx, speed_rate=None):
        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]
        if speed_rate:
            wav = librosa.effects.time_stretch(wav, speed_rate)
        if len(wav) < self.sr:
            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')
        return wav[:self.sr]

    def get_silent_wav(self, num_noise=1, max_ratio = 0.5):
        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)

    def timeshift(self, wav, ms = 100):
        shift = (self.sr * ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(wav, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def get_noisy_wav(self, idx):
        scale = random.uniform(0.75, 1.25)
        num_noise = random.choice([1,2])
        max_ratio = random.choice([0.1, 0.5, 1, 2])
        mix_noise_proba = 0.25
        shift_range = random.randint(80, 120)
        if random.random() < mix_noise_proba:
            return scale * (self.timeshift(self.get_one_word_wav(idx), shift_range) + self.get_mix_noises(
                num_noise, max_ratio))
        else:
            return scale * self.timeshift(self.get_one_word_wav(idx), shift_range)

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        """reads one sample"""
        wav_numpy = self.preprocess_fun(self.get_noisy_wav(idx), **self.preprocess_param)
        if self.resize_shape:
            wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)
        wav_tensor = torch.from_numpy(wav_numpy).float()
        if not self.is_1d:
            wav_tensor = wav_tensor.unsqueeze(0)

        label_word = self.semi_dict[self.wav_list[idx].split('/')[-1]]
        if label_word == "unknown":
            label = 10
        elif label_word == 'silence':
            label = 11
        else:
            label = self.label_words_dict[label_word]

        return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}