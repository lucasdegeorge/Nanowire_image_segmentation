#%% Libraries 

import torch
import torch.nn.functional as F
from torchvision import transforms
import json
import sys
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
from torchsummary import summary

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

from dataloader import * 

#%% dataloarder 
in_channels = arguments["model"]["in_channels"]
train_labeled_dataloader, eval_labeled_dataloader,  unlabeled_dataloader = get_dataloaders(in_channels, batch_size=batch_size)

#%% micro - Dataloaders for tests :  ## Does not work since updates in dataloader.py

in_channels = arguments["model"]["in_channels"]

# micro paths 
micro_labeled_image_dir = "C:/Users/lucas.degeorge/Documents/Images/other_images/micro_batch_for_tests/labeled_images"
micro_masks_dir = "C:/Users/lucas.degeorge/Documents/Images/other_images/micro_batch_for_tests/binary_masks"
micro_unlabeled_image_dir = "C:/Users/lucas.degeorge/Documents/Images/other_images/micro_batch_for_tests/unlabeled_images"
micro_folder_where_write = "C:/Users/lucas.degeorge/Documents/Images/other_images/micro_batch_for_tests"


train_images, eval_images, train_masks, eval_masks = load_labeled_data(in_channels, micro_labeled_image_dir, micro_masks_dir, folder_where_write=micro_folder_where_write)
micro_labeled_dataset = train_LabeledDataset(train_images, train_masks, transform=None)
micro_unlabeled_dataset = UnlabeledDataset(in_channels, micro_unlabeled_image_dir, transform=None, folder_where_write=micro_folder_where_write)

micro_labeled_dataloader = torch.utils.data.DataLoader(micro_labeled_dataset, batch_size=2, shuffle=True, drop_last=True)
micro_unlabeled_dataloader = torch.utils.data.DataLoader(micro_unlabeled_dataset, batch_size=2, shuffle=True)

#%% Trainer unit tests 

from trainer import *

# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/test_trainer_{}'.format(timestamp))

# # mode super
model_test = Model(mode='semi')
model_test.to(device)

# summary(model_test.cuda(), (3, 1024, 1024))

trainer_test = Trainer(model_test, micro_labeled_dataloader, micro_unlabeled_dataloader, micro_labeled_dataloader)  ## Just for test : here same data in train and eval
# trainer_test.train_super_1epoch(0, writer)
# trainer_test.train_semi_1epoch(0, writer)
# trainer_test.eval_1epoch(0)
trainer_test.train()

#%% Accuracy unit tests 

from inference_scores import * 

# tests
# for one images
mask_path = "C:/Users/lucas.degeorge/Documents/Images/binary_masks/0000021_mask.png"
predict = mask_converter(mask_path, out="one-hot")
target = T.functional.to_tensor(Image.open(mask_path).convert("L")) * 2.5
target = target.to(torch.uint8) 

# print(mIoU(predict, target, batch=False))

# for a batch
targets = []
predicts = []
for filename in os.listdir(masks_dir):
    if filename.endswith(filetype):
        predict = mask_converter(mask_path, out="one-hot")
        target = T.functional.to_tensor(Image.open(mask_path).convert("L")) * 2.5
        target = target.to(torch.uint8) 
        predicts.append(predict)
        targets.append(target)

# mask_tests_dataset = train_LabeledDataset(predicts, targets, transform=None)
# mask_tests_dataloader = torch.utils.data.DataLoader(mask_tests_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

# for predict, target in mask_tests_dataloader:
#     print(mIoU(predict, target[:,0], 3, True))

model = Model(mode="semi")

for image, mask in eval_labeled_dataloader:
    image = image.to(device)
    pred = model(image, eval=True)["output_l"]
    mask = one_hot_to_image(mask.permute(0,2,3,1), class_values=[0,1,2])
    mask = mask.to(device)
    print(mIoU(pred, mask, 3, True))


#%% Resnet unit tests 

from model.resnet import *

# image size in our dataset
image = Image.open("C:/Users/lucas.degeorge/Documents/Images/labeled_images/l_000001.png").convert("L")
convert_tensor = transforms.ToTensor()
img = convert_tensor(image) 
img = torch.unsqueeze(img, dim=0)

# image size as in ImageNet 
# image1 = image.resize((224,224))
# img1 = convert_tensor(image1) 
# img1 = torch.unsqueeze(img1, dim=0)

rn18 = ResNet50_bb(arguments)
res = rn18(img)

for i in range(len(res)):
    print(i, res[i].shape)


#%% Encoder unit tests 

from model.encoder import * 

print("resnet", 50)
enc = Encoder(arguments)
for image, mask in micro_labeled_dataloader:
    res = enc(image)
    print(res.shape)
    break


#%% Decoder unit tests

from model.encoder import * 
from model.decoders import *

upscale = 8
num_out_ch = 512
decoder_in_ch = num_out_ch // 4

nb_classes = arguments["model"]["nb_classes"]

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
main_dec = MainDecoder(upscale, decoder_in_ch, num_classes=nb_classes)
for name, decoder in aux_decoder_dict.items():
    print(name)
    dec = decoder(upscale, decoder_in_ch, num_classes=nb_classes)
    for image, mask in micro_labeled_dataloader:
        res = enc(image)
        pred = main_dec(res)
        res = dec(res, pred)
        print(res.shape)
        res = F.interpolate(res, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=True)
        print(res.shape)
        break

#%% Model unit tests and criterion (DICE)

from model.model import * 
from monai.losses import DiceLoss, DiceCELoss

# # mode super
# model_test = Model(mode='super')
# for image, mask in labeled_dataloader:
#     res = model_test(image, None)
#     break

# mode semi
model_test = Model(mode='semi')
dataloader = iter(zip(cycle(micro_labeled_dataloader), micro_unlabeled_dataloader))

for (x_l, target_l), x_ul in dataloader:
    x_l = x_l.to(device)
    target_l = target_l.to(device)
    x_ul = x_ul.to(device)

    res = model_test(x_l, x_ul=x_ul)
    out_l, out_ul = res["output_l"], res["output_ul"]
    print(out_l.shape)
    criterion = DiceCELoss(reduction='mean')
    loss = criterion(out_l, target_l)
    break
