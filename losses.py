#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from monai.losses import DiceLoss, DiceCELoss

#%% Supervised losses 

def supervised_loss(input, target, mode):
    assert input.requires_grad == True and target.requires_grad == False, "Error in requires_grad - supervised"
    assert input.size() == target.size(), "Input and target must have the same size, must be (batch_size * num_classes * H * W)"

    if mode == "CE":
        return F.cross_entropy(input, target)
    elif mode == "DICE":
        dice_loss = DiceLoss(reduction='mean')
        return dice_loss(input, target)
    elif mode == "DICE-CE":
        diceCE = DiceCELoss(reduction='mean')
        return diceCE(input, target) 
    else:
        ValueError("Invalid value for mode. Must be in ['CE', 'DICE', DICE-CE]")


#%% Eval loss 

def eval_loss(input, target, mode):
    assert input.requires_grad == False and target.requires_grad == False, "Error in requires_grad - eval"
    assert input.size() == target.size(), "Input and target must have the same size, must be (batch_size * num_classes * H * W)"

    if mode == "CE":
        return F.cross_entropy(input, target)
    elif mode == "DICE":
        dice_loss = DiceLoss(reduction='mean')
        return dice_loss(input, target)
    elif mode == "DICE-CE":
        diceCE = DiceCELoss(reduction='mean')
        return diceCE(input, target) 
    else:
        ValueError("Invalid value for mode. Must be in ['CE', 'DICE', DICE-CE]")


#%% unsupervised losses 

def unsupervised_loss(input, target, mode):
    assert input.requires_grad == True and target.requires_grad == False, "Error in requires_grad"
    assert input.size() == target.size(), "Input and target must have the same size, must be (batch_size * num_classes * H * W)"

    if mode == "dice":
        dice_loss = DiceLoss(reduction='mean')
        return dice_loss(input, target)
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
