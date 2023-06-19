#%% 
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import json

# Device configuration
with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

in_channels = 1
num_classes = 3

#%% 

# residual block for the resnet 18 and 34 
class old_ResidualBlock_2sl(nn.Module): 
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None, dilation=1):
        super(old_ResidualBlock_2sl, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias=False),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    

# residual block for the resnet 50, 101 and 152 
class old_ResidualBlock_3sl(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=1):
        super(old_ResidualBlock_3sl, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*self.expansion) 
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class old_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=num_classes, isDilation=True, dilate_scale=8, multi_grid=(1, 2, 4)):
        super(old_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(   # Ici il faut regarder à quoi correspond deep_base, quel est l'intérêt et pourquoi on utilise cela
            nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if isDilation:
            self.layer0 = self._make_layer(block, 64, layers[0], stride=1, dilation=multi_grid[0])
            self.layer1 = self._make_layer(block, 128, layers[1], stride=2, dilation=dilate_scale // multi_grid[1])
            self.layer2 = self._make_layer(block, 256, layers[2], stride=2, dilation=dilate_scale // multi_grid[2])
            self.layer3 = self._make_layer(block, 512, layers[3], stride=2, dilation=dilate_scale)
        else:
            self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
            self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
            self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
            self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(32, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, nb_blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:  # if the dim of the residual does no match the dim of the output
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes*block.expansion
        for i in range(1, nb_blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

class old_ResnetBackbone(nn.Module):
    def __init__(self, orig_resnet):
        super(old_ResnetBackbone, self).__init__()

        self.num_features = 2048

        # Take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.maxpool = orig_resnet.maxpool
        self.layer0 = orig_resnet.layer0
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        tuple_features.append(x)
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)

        return tuple_features
    

def old_ResNet18_bb(isDilation = True):
    return old_ResnetBackbone(old_ResNet(old_ResidualBlock_2sl, [3,2,2,2], isDilation=isDilation))
    # return old_ResNet(old_ResidualBlock_2sl, [3,2,2,2], isDilation=isDilation)

def old_ResNet34_bb(isDilation = True):
    return old_ResnetBackbone(old_ResNet(old_ResidualBlock_2sl, [3,4,6,3], isDilation=isDilation))
    # return old_ResNet(old_ResidualBlock_2sl, [3,4,6,3], isDilation=isDilation)

def old_ResNet50_bb(isDilation = True):
    return old_ResnetBackbone(old_ResNet(old_ResidualBlock_3sl, [3,4,6,3], isDilation=isDilation))
    # return old_ResNet(old_ResidualBlock_3sl, [3,4,6,3], isDilation=isDilation)

def old_ResNet101_bb(isDilation = True):
    return old_ResnetBackbone(old_ResNet(old_ResidualBlock_3sl, [3,4,23,3], isDilation=isDilation))
    # return old_ResNet(old_ResidualBlock_3sl, [3,4,23,3], isDilation=isDilation)

def old_ResNet152_bb(isDilation = True):
    return old_ResnetBackbone(old_ResNet(old_ResidualBlock_3sl, [3,8,36,3], isDilation=isDilation))
    # return old_ResNet(old_ResidualBlock_3sl, [3,8,36,3], isDilation=isDilation)

old_resnet_bbs = {18 : old_ResNet18_bb, 
           34 : old_ResNet34_bb, 
           50 : old_ResNet50_bb, 
           101 : old_ResNet101_bb,
           152 : old_ResNet152_bb
           }