#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Tests

folder_where_write = "C:/Users/lucas.degeorge/Documents/Images"
masks = torch.load(folder_where_write + "/" + "binary_masks.pt")



#%% 

def masks_to_onehots(mask):
