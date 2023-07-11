#%% 

# most of the code from: https://github.com/akanametov/pix2pix/blob/main/gan/criterion.py

import torch
from torch import nn

class GeneratorLoss(nn.Module):
    def __init__(self, alpha1=100, alpha2=200):
        super().__init__()
        self.alpha1=alpha1
        self.alpha2=alpha2
        self.bce=nn.BCEWithLogitsLoss()
        self.l1=nn.L1Loss()
        
    def forward(self, fake, real, fake_pred):
        fake_target = torch.ones_like(fake_pred)

        real_image = real[:,0]
        real_mask = real[:,1]
        fake_image = fake[:,0]
        fake_mask = fake[:,1]

        loss = self.bce(fake_pred, fake_target) + self.alpha1*self.l1(fake_image, real_image) + self.alpha2*self.l1(fake_mask, real_mask)
        return loss
    
    
class DiscriminatorLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.loss_fn(fake_pred, fake_target)
        real_loss = self.loss_fn(real_pred, real_target)
        loss = (fake_loss + real_loss)/2
        return loss