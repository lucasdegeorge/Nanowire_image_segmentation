#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from dataloader import * 

#%% Supervised losses 

def supervised_loss(input, target, mode="CE"):
    assert input.requires_grad == True and target.requires_grad == False, "Error in requires_grad"
    assert input.size() == target.size(), "Input and target must have the same size, must be (batch_size * num_classes * H * W)"

    if mode == "CE":
        return F.cross_entropy(input, target)
    elif mode == "DICE":
        dice_loss = MulticlassDiceLoss(input.shape[2])
        return dice_loss(input, target)
    else:
        ValueError("Invalid value for mode. Must be in ['CE', 'DICE']")

    

class MulticlassDiceLoss(nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """
    def __init__(self, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """
        probabilities = logits
        intersection = (targets * probabilities).sum()
        
        mod_a = intersection.sum()
        mod_b = targets.numel()
        
        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = - dice_coefficient.log()
        # return 1 - dice_coefficient
        return dice_loss



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


#%% Weight ramp up for unsupervised loss 


def weight_ramp_up(current_step, rampup_length, max_value):
    """Generates the value of 'w' based on a sigmoid ramp-up curve."""
    if rampup_length == 0:
        return max_value
    else:
        current_step = max(0.0, min(current_step, rampup_length))
        phase = 1.0 - current_step / rampup_length
        value = max_value * (math.exp(-5.0 * phase * phase))
        return value
