#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import random
import uniform
import sys

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation")

from parameters import *
from encoder import * 
from decoders import *

#%% Model

class Model(nn.Module):
    def __init__(self, loss, mode='semi', arguments=arguments, upscale=8, aux_decoders_args=[1,1,1,1,1,1,1]):
        super(Model, self).__init__()

        self.mode = mode
        self.nb_classes = arguments["nb_classes"]

        # encoder
        self.nb_RNlayers = arguments["nb_RNlayers"]
        self.isDilation = arguments["isDilation"]
        if self.nb_RNlayers in [50, 101, 152]: self.in_channels_psp = 2048
        elif self.nb_RNlayers in [18, 34]: self.in_channels_psp = 512
        else: raise ValueError("invalid nb_RNlayers")

        # decoders
        self.upscale = arguments["upscale"]
        self.in_channels_dec = self.in_channels_psp // 4

        # losses
        self.loss = loss #### Here change #### 

        # Model
        self.encoder = Encoder(nb_RNlayers=self.nb_RNlayers, in_channls_psp=self.in_channels_psp, isDilation=self.isDilation)
        self.main_decoder = MainDecoder(self.upscale, self.in_channels_dec, self.nb_classes)

        if self.mode == "semi":
            vat_decoder = [VATDecoder(self.upscale, self.in_channels_dec, self.nb_classes, xi=arguments['xi'],eps=arguments['eps']) for _ in range(arguments['VATDecoder'])]
            drop_decoder = [DropOutDecoder(self.upscale, self.in_channels_dec, self.nb_classes,drop_rate=arguments['drop_rate'], spatial_dropout=arguments['spatial']) for _ in range(arguments['DropOutDecoder'])]
            cut_decoder = [CutOutDecoder(self.upscale, self.in_channels_dec, self.nb_classes, erase=arguments['erase']) for _ in range(arguments['CutOutDecoder'])]
            context_m_decoder = [ContextMaskingDecoder(self.upscale, self.in_channels_dec, self.nb_classes) for _ in range(arguments['ContextMaskingDecoder'])]
            object_masking = [ObjectMaskingDecoder(self.upscale, self.in_channels_dec, self.nb_classes) for _ in range(arguments['ObjectMaskingDecoder'])]
            feature_drop = [FeatureDropDecoder(self.upscale, self.in_channels_dec, self.nb_classes) for _ in range(arguments['FeatureDropDecoder'])]
            feature_noise = [FeatureNoiseDecoder(self.upscale, self.in_channels_dec, self.nb_classes, uniform_range=arguments['uniform_range']) for _ in range(arguments['FeatureNoiseDecoder'])]

            self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder, *context_m_decoder, *object_masking, *feature_drop, *feature_noise])

    def forward(self, x_l, x_ul=None, tgt_l=None, tgt_ul=None):

        output_l = self.main_decoder(self.encoder(x_l))
        if output_l.shape != x_l.shape:
            output_l = F.interpolate(output_l, size=x_l.shape, mode='bilinear', align_corners=True)

        if self.mode == 'super':
            return {"output_l" :  output_l}
        
        elif self.mode == 'semi':
            # Prediction by main decoder 
            x_ul = self.encoder(x_ul)
            output_ul = self.main_decoder(x_ul)
            # Prediction by auxiliary decoders
            outputs_ul = [aux_decoder(x_ul, output_ul.detach()) for aux_decoder in self.aux_decoders]
            outputs_ul = [F.interpolate(output, size=x_l.shape, mode='bilinear', align_corners=True) for output in outputs_ul if output.shape != x_ul.shape]
            
            return {"output_l" : output_l, "outputs_ul" : outputs_ul}
        






        