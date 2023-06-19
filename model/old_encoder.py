#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from old_resnet import *

# Device configuration
with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

#%% 

class old_PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(old_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output
    

class old_Encoder(nn.Module):
    def __init__(self, nb_RNlayers=50, in_channels_psp=2048, isDilation=True):
        super(old_Encoder, self).__init__()

        model = old_resnet_bbs[nb_RNlayers](isDilation=isDilation)

        self.base = nn.Sequential(
            nn.Sequential(model.conv1, model.maxpool),
            model.layer0,
            model.layer1,
            model.layer2,
            model.layer3
        )
        self.psp = old_PSPModule(in_channels_psp, bin_sizes=[1, 2, 3, 6])

    def forward(self, x):
        x = self.base(x)
        x = self.psp(x)
        return x

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()