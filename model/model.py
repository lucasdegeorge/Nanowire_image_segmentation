#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import itertools
import json

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation")

from encoder import * 
from decoders import *
from preprocessing.dataloader import * 

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    model_arguments = arguments["model"]


#%% Model

class Model(nn.Module):
    def __init__(self, mode='semi', arguments=model_arguments):
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

        # Model
        self.encoder = Encoder(nb_RNlayers=self.nb_RNlayers, in_channels_psp=self.in_channels_psp, isDilation=self.isDilation)
        self.main_decoder = MainDecoder(self.upscale, self.in_channels_dec, self.nb_classes)

        if self.mode == "semi":
            drop_decoder = [DropOutDecoder(self.upscale, self.in_channels_dec, self.nb_classes,drop_rate=arguments['drop_rate'], spatial_dropout=arguments['spacial_dropout']) for _ in range(arguments['DropOutDecoder'])]
            feature_drop = [FeatureDropDecoder(self.upscale, self.in_channels_dec, self.nb_classes) for _ in range(arguments['FeatureDropDecoder'])]
            feature_noise = [FeatureNoiseDecoder(self.upscale, self.in_channels_dec, self.nb_classes, uniform_range=arguments['uniform_range']) for _ in range(arguments['FeatureNoiseDecoder'])]
            vat_decoder = [VATDecoder(self.upscale, self.in_channels_dec, self.nb_classes, xi=arguments['xi'],eps=arguments['eps']) for _ in range(arguments['VATDecoder'])]
            cut_decoder = [CutOutDecoder(self.upscale, self.in_channels_dec, self.nb_classes, erase=arguments['erase']) for _ in range(arguments['CutOutDecoder'])]
            context_m_decoder = [ContextMaskingDecoder(self.upscale, self.in_channels_dec, self.nb_classes) for _ in range(arguments['ContextMaskingDecoder'])]
            object_masking = [ObjectMaskingDecoder(self.upscale, self.in_channels_dec, self.nb_classes) for _ in range(arguments['ObjectMaskingDecoder'])]

            self.aux_decoders = nn.ModuleList([*drop_decoder, *feature_drop, *feature_noise, *vat_decoder, *cut_decoder, *context_m_decoder, *object_masking])

    def forward(self, x_l, x_ul=None):

        output_l = self.main_decoder(self.encoder(x_l))
        if output_l.shape != x_l.shape:
            output_l = F.interpolate(output_l, size=(x_l.size(2), x_l.size(3)), mode='bilinear', align_corners=True)

        if self.mode == 'super':
            return {"output_l" :  output_l}
        
        elif self.mode == 'semi':
            assert x_ul is not None
            # Prediction by main decoder 
            inter_ul = self.encoder(x_ul)
            output_ul = self.main_decoder(inter_ul)

            # Prediction by auxiliary decoders
            aux_outputs_ul = [aux_decoder(inter_ul, output_ul.detach()) for aux_decoder in self.aux_decoders]
            aux_outputs_ul = [F.interpolate(output, size=(x_ul.size(2), x_ul.size(3)), mode='bilinear', align_corners=True) for output in aux_outputs_ul if output.shape != x_ul.shape]

            output_ul = F.interpolate(output_ul, size=(x_ul.size(2), x_ul.size(3)), mode='bilinear', align_corners=True)

            return {"output_l" : output_l, "output_ul" : output_ul, "aux_outputs_ul" : aux_outputs_ul}
        
    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        if self.mode == 'semi':
            return itertools.chain(self.encoder.get_module_params(), self.main_decoder.parameters(), 
                        self.aux_decoders.parameters())

        return itertools.chain(self.encoder.get_module_params(), self.main_decoder.parameters())

       