#%% Libraries 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import json
import sys
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/model")

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)

from preprocessing.dataloader import * 

in_channels = 1
num_classes = 3

#%% Dataloaders for tests : 







#%% Trainer unit tests 

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/test_trainer_{}'.format(timestamp))

# # mode super
model_test = Model(mode='super')

trainer_test = Trainer(model_test, labeled_dataloader, unlabeled_dataloader, labeled_dataloader)
# trainer_test.train_super_1epoch(0, writer)
# trainer_test.train_semi_1epoch(0, writer)
# trainer_test.eval_1epoch(0)
trainer_test.train()



#%% Resnet unit tests 

from model.resnet import *

image = Image.open("C:/Users/lucas.degeorge/Documents/Images/labeled_images/0000001.png")#.convert("RGB")
image1 = image.resize((224,224))

convert_tensor = transforms.ToTensor()
img = convert_tensor(image)   # image size in our dataset
img1 = convert_tensor(image1) # image size as in ImageNet 
img = torch.unsqueeze(img, dim=0)
img1 = torch.unsqueeze(img1, dim=0)
img1.shape

rn18 = ResNet34_bb()
res = rn18(img)

for i in range(len(res)):
    print(i, res[i].shape)


#%% Encoder unit tests 

from model.encoder import * 

for x,y in [(18,512)]:
    print("resnet", x)
    enc = Encoder(nb_RNlayers=x)
    for image, mask in labeled_dataloader:
        res = enc(image)
        print(res.shape)
        break


#%% Decoder unit tests

from model.encoder import * 
from model.decoders import *

upscale = 8
num_out_ch = 512
decoder_in_ch = num_out_ch // 4

# enc = Encoder(nb_RNlayers=18, in_channels_psp=512)
# dec = FeatureNoiseDecoder(upscale, decoder_in_ch, num_classes=num_classes)
# for image, mask in labeled_dataloader:
#     # print(image.shape)
#     res = enc(image)
#     # print(res.shape)
#     res = dec(res)
#     print(res.shape)
#     res = F.interpolate(res, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=True)
#     print(res.shape)
#     break

enc = Encoder(nb_RNlayers=18, in_channels_psp=512)
main_dec = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)
for name, decoder in aux_decoder_dict.items():
    print(name)
    dec = decoder(upscale, decoder_in_ch, num_classes=num_classes)
    for image, mask in labeled_dataloader:
        res = enc(image)
        pred = main_dec(res)
        res = dec(res, pred)
        print(res.shape)
        res = F.interpolate(res, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=True)
        print(res.shape)
        break

#%% Model unit tests 

from model import * 

# # mode super
# model_test = Model(mode='super')
# for image, mask in labeled_dataloader:
#     res = model_test(image, None)
#     break

# mode semi
model_test = Model(mode='semi')
# dataloader = iter(zip(cycle(labeled_dataloader), unlabeled_dataloader))
for x_ul in unlabeled_dataloader:
    for x_l, _ in labeled_dataloader:
        res = model_test(x_l, x_ul=x_ul)
        break
    break

dataloader = iter(zip(cycle(labeled_dataloader), unlabeled_dataloader))
for (x_l, mask), x_ul in dataloader:
    res = model_test(x_l, x_ul=x_ul)
    break
