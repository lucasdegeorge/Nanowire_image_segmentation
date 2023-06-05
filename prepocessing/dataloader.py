#%% 

import os
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt 
import numpy as np
import cv2 

batch_size = 32 

labeled_image_dir = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
masks_dir = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"
unlabeled_image_dir = "C:/Users/lucas.degeorge/Documents/Images/unlabeled_images"

filetype = '.png'

#%% 

# Labeled data and masks

def load_labeled_data(image_dir, annotation_dir):
    labeled_images = []
    masks = []
    for filename in os.listdir(image_dir):
        if filename.endswith(filetype):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(annotation_dir, filename[:-4] + '_mask.png')
            if os.path.isfile(mask_path):
                labeled_images.append(image_path)
                masks.append(mask_path)
    return labeled_images, masks


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images, self.masks = load_labeled_data(image_dir, annotation_dir)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index]).convert('L')
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
    
    def __len__(self):
        return len(self.images)



# Unlabeled data 

def load_unlabeled_data(image_dir):
    unlabeled_images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(filetype):
            image_path = os.path.join(image_dir, filename)
            unlabeled_images.append(image_path)
    return unlabeled_images


class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = load_unlabeled_data(image_dir)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.images)



# Definition 

labeled_dataset = LabeledDataset(labeled_image_dir, masks_dir, transform=None)
unlabeled_dataset = UnlabeledDataset(unlabeled_image_dir, transform=None)

labeled_dataloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)


#%% Display images and masks separated

def display_image_with_mask(image, mask):
    fig, ax = plt.subplots(1, 2)
    
    ax[0].imshow(image) # Display the image
    ax[0].set_title('Image')
    ax[1].imshow(mask, cmap='gray')     # Display the mask
    ax[1].set_title('Mask')
    
    for axis in ax:
        axis.axis('off')
    
    plt.tight_layout()
    plt.show()

# Test : 
image, mask = labeled_dataset[1]
display_image_with_mask(image, mask)


#%% Display image and mask overlayed

def display_image_mask_overlayed(image, mask, alpha=0.6):
    image = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    mask = np.array(mask.getdata()).reshape(mask.size[0], mask.size[1])

    color_map = {
        0: [0, 0, 0],     # Background - Black
        127: [255, 0, 0], # Droplet - Red
        255: [0, 255, 0]  # Wire - Green
    }

    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for value, color in color_map.items():
        colored_mask[mask == value] = color
    overlay = cv2.addWeighted(image.astype(np.uint8), 0.7, colored_mask.astype(np.uint8), 0.3, 0)

    _, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis('off')
    plt.show()

# Tests : 
image, mask = labeled_dataset[78]
display_image_mask_overlayed(image, mask)