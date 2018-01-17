import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
from glob import glob
import random
from skimage.transform import resize
import pandas as pd
from random import sample

SR=16000

class SpeechDataset(Dataset):
    def __init__(self, mode, label_words_dict, wav_list, add_noise, preprocess_fun, preprocess_param = {}, sr=SR, resize_shape=None, is_1d=False):
        """Args:
                mode: train or evaluate or test
                label_words_dict: a dict of words for labels
                wav_list: a list of wav file paths
                add_noise: boolean. if background noise should be added
                preprocess_fun: function to load/process wav file
                preprocess_param: params for preprocess_fun
                sr: default 16000
                resize_shape: None. only for 2d cnn.
                is_1d: boolean. if it is going to be 1d cnn or 2d cnn
        """
        self.mode = mode
        self.label_words_dict = label_words_dict
        self.wav_list = wav_list
        self.add_noise = add_noise
        self.sr = sr
        self.n_silence = int(len(wav_list) * 0.09)
        self.preprocess_fun = preprocess_fun
        self.preprocess_param = preprocess_param

        # read all background noise here
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("../input/train/audio/_background_noise_/*.wav")]
        self.resize_shape = resize_shape
        self.is_1d = is_1d

    def get_one_noise(self):
        """generates one single noise clip"""
        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]
        # only takes out 16000
        start_idx = random.randint(0, len(selected_noise) - 1 - self.sr)
        return selected_noise[start_idx:(start_idx + self.sr)]

    def get_mix_noises(self, num_noise=1, max_ratio=0.1):
        result = np.zeros(self.sr)
        for _ in range(num_noise):
            result += random.random() * max_ratio * self.get_one_noise()
        return result / num_noise if num_noise > 0 else result

    def get_one_word_wav(self, idx):
        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]
        if len(wav) < self.sr:
            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')
        return wav[:self.sr]

    def get_silent_wav(self, num_noise=1, max_ratio=0.5):
        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)

    def timeshift(self, wav, ms=100):
        shift = (self.sr * ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(wav, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def get_noisy_wav(self, idx):
        scale = random.uniform(0.75, 1.25)
        num_noise = random.choice([1, 2])
        max_ratio = random.choice([0.1, 0.5, 1, 1.5])
        mix_noise_proba = random.choice([0.1, 0.3])
        shift_range = random.randint(80, 120)
        one_word_wav = self.get_one_word_wav(idx)
        if random.random() < mix_noise_proba:
            return scale * (self.timeshift(one_word_wav, shift_range) + self.get_mix_noises(
                num_noise, max_ratio))
        else:
            return one_word_wav

    def __len__(self):
        if self.mode == 'test':
            return len(self.wav_list)
        else:
            return len(self.wav_list) + self.n_silence

    def __getitem__(self, idx):
        """reads one sample"""
        if idx < len(self.wav_list):
            wav_numpy = self.preprocess_fun(
                self.get_one_word_wav(idx) if self.mode != 'train' else self.get_noisy_wav(idx),
                **self.preprocess_param)
            if self.resize_shape:
                wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)
            wav_tensor = torch.from_numpy(wav_numpy).float()
            if not self.is_1d:
                wav_tensor = wav_tensor.unsqueeze(0)
            if self.mode == 'test':
                return {'spec': wav_tensor, 'id': self.wav_list[idx]}

            label = self.label_words_dict[self.wav_list[idx].split("/")[-2]] if self.wav_list[idx].split(
                "/")[-2] in self.label_words_dict else len(self.label_words_dict)

            return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}

        else:
            """generates silence here"""
            wav_numpy = self.preprocess_fun(self.get_silent_wav(
                num_noise=random.choice([0, 1, 2, 3]),
                max_ratio=random.choice([x / 10. for x in range(20)])), **self.preprocess_param)
            if self.resize_shape:
                wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)
            wav_tensor = torch.from_numpy(wav_numpy).float()
            if not self.is_1d:
                wav_tensor = wav_tensor.unsqueeze(0)
            return {'spec': wav_tensor, 'id': 'silence', 'label': len(self.label_words_dict) + 1}


def get_label_dict():
    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    label_to_int = dict(zip(words, range(len(words))))
    int_to_label = dict(zip(range(len(words)), words))
    int_to_label.update({len(words): 'unknown', len(words) + 1: 'silence'})
    return label_to_int, int_to_label


def get_wav_list(words, unknown_ratio=0.2):
    full_train_list = glob("../input/train/audio/*/*.wav")
    full_test_list = glob("../input/test/audio/*.wav")

    # sample full train list
    sampled_train_list = []
    for w in full_train_list:
        l = w.split("/")[-2]
        if l not in words:
            if random.random() < unknown_ratio:
                sampled_train_list.append(w)
        else:
            sampled_train_list.append(w)

    return sampled_train_list, full_test_list


def get_sub_list(num, sub_path):
    lst = []
    df = pd.read_csv(sub_path)
    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
    each_num = int(num * 0.085)
    for w in words:
        tmp = df['fname'][df['label'] == w].sample(each_num).tolist()
        lst += ["../input/test/audio/" + x for x in tmp]
    return lst


def get_semi_list(words, sub_path, unknown_ratio=0.2, test_ratio=0.2):
    train_list, _ = get_wav_list(words=words, unknown_ratio=unknown_ratio)
    test_list = get_sub_list(num=int(len(train_list) * test_ratio), sub_path=sub_path)
    lst = train_list + test_list
    return sample(lst, len(lst))


def preprocess_mfcc(wave):

    spectrogram = librosa.feature.melspectrogram(wave, sr=SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    idx = [spectrogram > 0]
    spectrogram[idx] = np.log(spectrogram[idx])

    dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
    mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
    mfcc = np.hstack(mfcc)
    mfcc = mfcc.astype(np.float32)
    return mfcc


def preprocess_mel(data, n_mels=40, normalization=False):
    spectrogram = librosa.feature.melspectrogram(data, sr=SR, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    if normalization:
        spectrogram = spectrogram.spectrogram()
        spectrogram -= spectrogram
    return spectrogram


def preprocess_wav(wav, normalization=True):
    data = wav.reshape(1, -1)
    if normalization:
        mean = data.mean()
        data -= mean
    return data