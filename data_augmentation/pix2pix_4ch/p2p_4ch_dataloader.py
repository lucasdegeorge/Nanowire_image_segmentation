#%% 
import os
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
import json

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

c2n = True

if c2n:
    image_folder = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
    mask_folder = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"
else:
    image_folder = "C:/Users/lucas/Desktop/labeled_images"
    mask_folder = "C:/Users/lucas/Desktop/binary_masks"

filetype = '.png'

#%% 

class ImageMaskDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        # self.images = [ t.to(device) for t in images]
        # self.masks = [ t.to(device) for t in masks]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
    

def create_dataloader(image_folder, mask_folder, batch_size, shuffle=True, pin_memory=True):

    # load and convet images
    converter = T.ToTensor()
    images = []
    masks = []
    # load and converter images 
    for filename in os.listdir(image_folder):
        if filename.endswith(filetype):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("L")
            image = converter(image)
            images.append(image)

            if mask_folder is not None:
                mask_path = os.path.join(mask_folder, filename[:-4] + '_mask.png')
                if os.path.isfile(mask_path):
                    mask = Image.open(mask_path).convert("L")
                    mask = T.functional.to_tensor(mask) * 255
                    # mask = mask.to(torch.uint8) 
                    masks.append(mask)
    # return images, masks

    dataset = ImageMaskDataset(images, masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory)
    print("dataloader ok")
    return dataloader

# images, masks = create_dataloader(image_folder, mask_folder, batch_size=2, shuffle=True, pin_memory=True)

