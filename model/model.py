import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import random
import uniform

from encoder import * 
from decoders import *


class Model(nn.Module):
    def __init__(self, loss, mode='semi', nb_classes=3, nb_RNlayers=50, isDilation=True, upscale=8):
        super(Model, self).__init__()

        self.mode = mode
        self.nb_classes = nb_classes
        # encoder
        self.nb_RNlayers = nb_RNlayers
        self.isDilation = isDilation
        if nb_RNlayers in [50, 101, 152]: self.in_channels_psp = 2048
        elif nb_RNlayers in [18, 34]: self.in_channels_psp = 512
        else: raise ValueError("invalid nb_RNlayers")
        # decoders 
        self.upscale = upscale
        self.in_channels_dec = self.in_channels_psp // 4 
        # losses
        self.loss = loss #### Here change #### 

        # Model 
        self.encdoder = Encoder(nb_RNlayers=self.nb_RNlayers, in_channls_psp=self.in_channels_psp, isDilation=self.isDilation)
        self.main_decoder = MainDecoder(upscale=self.upscale, conv_in_ch=self.in_channels_dec, num_classes=self.nb_classes)