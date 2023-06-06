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

#%% Resnet unit tests 

from resnet import *

image = Image.open("C:/Users/lucas.degeorge/Documents/Images/labeled_images/0000001.png")
image1 = image.resize((224,224))

convert_tensor = transforms.ToTensor()
img = convert_tensor(image)   # image size in our dataset
img1 = convert_tensor(image1) # image size as in ImageNet 
img = torch.unsqueeze(img, dim=0)
img1 = torch.unsqueeze(img1, dim=0)
img1.shape

rn18 = ResNet50_bb()
res = rn18(img1)

