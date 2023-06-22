#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import * 

#%% Supervised losses 

def supervised_loss(input, target, mode="CE"):
    assert input.requires_grad == True and target.requires_grad == False, "Error in requires_grad"
    assert input.size() == target.size(), "Input and target must have the same size, must be (batch_size * num_classes * H * W)"

    if mode == "CE":
        return F.cross_entropy(input, target)
    else:
        ValueError("Invalid value for mode. Must be in ['CE']")



#%% unsupervised losses 

def unsupervised_loss(input, target, mode="mse"):
    assert input.requires_grad == True and target.requires_grad == False, "Error in requires_grad"
    assert input.size() == target.size(), "Input and target must have the same size, must be (batch_size * num_classes * H * W)"

    if mode == "mse":
        input = F.softmax(input, dim=1)
        return F.mse_loss(input, target, reduction='mean')
    if mode == "kl":
        input = F.log_softmax(input, dim=1)
        return F.kl_div(input, target, reduction='mean')
    if mode == "js":
        M = (F.softmax(input, dim=1) + target) * 0.5
        kl_P = F.kl_div(F.log_softmax(input, dim=1), M, reduction='mean')
        kl_Q = F.kl_div(torch.log(target + 1e-5), M, reduction='mean')
        return (kl_P + kl_Q) * 0.5
    else:
        raise ValueError("Invalid value for mode. Must be in ['mse', 'kl', 'js']")
