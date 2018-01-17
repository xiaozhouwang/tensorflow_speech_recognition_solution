import torch
import torch.nn.functional as F
import torch.nn as nn


class VGG1D(torch.nn.Module):
    def __init__(self, features):
        super(VGG1D, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 12),
        )

    def forward(self, x):
        x = self.features(x)
        max_pooled = F.adaptive_max_pool1d(x, 1)
        avg_pooled = F.adaptive_avg_pool1d(x, 1)
        x = torch.cat([max_pooled, avg_pooled], dim=1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGGMEL(torch.nn.Module):
    def __init__(self, features):
        super(VGGMEL, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(256*6, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 12),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=True, mk=2, ms=2, lk=5,ls=1,lp=2, in_c=1):
    layers = []
    in_channels = in_c
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=mk, stride=ms)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=lk, padding=lp, stride=ls)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg1d():
    return VGG1D(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 'M', 512, 'M', 1024, 'M', 1024, 'M', 512, 'M', 256, 'M']))


def vggmel():
    return VGGMEL(make_layers([64, 64, 'M', 512, 512,'M', 1024, 1024, 'M', 512, 256, 'M'], lk=3, lp=1, in_c=40))
