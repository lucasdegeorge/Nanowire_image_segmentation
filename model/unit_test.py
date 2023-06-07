#%% Libraries 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np
import sys

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation") 

from preprocessing.dataloader import * 

in_channels = 1
num_classes = 3

#%% Resnet unit tests 

from resnet import *

image = Image.open("C:/Users/lucas.degeorge/Documents/Images/labeled_images/0000001.png")#.convert("RGB")
image1 = image.resize((224,224))

convert_tensor = transforms.ToTensor()
img = convert_tensor(image)   # image size in our dataset
img1 = convert_tensor(image1) # image size as in ImageNet 
img = torch.unsqueeze(img, dim=0)
img1 = torch.unsqueeze(img1, dim=0)
img1.shape

rn18 = ResNet50_bb()
res = rn18(img)


#%% Encoder unit tests 

from encoder import * 

enc = Encoder()
for image, mask in labeled_dataloader:
    res = enc(image)
    print("step")


#%% Decoder unit tests

from encoder import * 
from decoders import *

upscale = 8
num_out_ch = 2048
decoder_in_ch = num_out_ch // 4

enc = Encoder()
dec = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)
for image, mask in labeled_dataloader:
    print(image.shape)
    res = enc(image)
    print(res.shape)
    res = dec(res)
    print(res.shape)
    res = F.interpolate(res, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=True)
    print(res.shape)
    break

