#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import itertools

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation") 

from preprocessing.dataloader import * 
from parameters import * 

#%% 

criterion = nn.MSELoss(reduction='mean') 

input = torch.randn(2,2,2, 2, requires_grad=True)
target = torch.randn(2,2,2, 2)
output = criterion(input, target)