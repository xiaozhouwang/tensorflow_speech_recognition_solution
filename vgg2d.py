import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class VGG(torch.nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        n_labels = 12
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(256, n_labels),
        )

    def forward(self, x):
        x = self.features(x)
        max_x = F.adaptive_max_pool2d(x, (1, 1))
        avg_x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.cat([max_x, avg_x], 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=True, ks=3, kp=1):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=ks, padding=kp)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg2d():
    """VGG 16-layer model with batch normalization"""
    return VGG(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 256, 256, 256, 'M']))
