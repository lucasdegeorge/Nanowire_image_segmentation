#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import itertools

#%% 

criterion = nn.MSELoss(reduction='mean') 

input = torch.randn(2,2,2, 2, requires_grad=True)
target = torch.randn(2,2,2, 2)
output = criterion(input, target)