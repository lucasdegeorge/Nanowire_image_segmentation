#%% 
import os
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import cv2

#%% 

def load_data(image_dir, annotation_dir):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(annotation_dir, filename[:-4] + '.png')
            if os.path.isfile(mask_path):
                images.append(image_path)
                masks.append(mask_path)
    return images, masks

class PascalVOCDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images, self.masks = load_data(image_dir, annotation_dir)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index]).convert('L')
        
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask
    
    def __len__(self):
        return len(self.images)
    
#%% Tets 

image_dir = "C:/Users/lucas.degeorge/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
annotation_dir = "C:/Users/lucas.degeorge/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass"

dataset = PascalVOCDataset(image_dir, annotation_dir, transform=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


#%% Display images and masks separated

def display_image_with_mask(image, mask):
    # Create a figure and axes
    fig, ax = plt.subplots(1, 2)
    
    # Display the image
    ax[0].imshow(image)
    ax[0].set_title('Image')
    
    # Display the mask
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    
    for axis in ax:
        axis.axis('off')
    
    plt.tight_layout()
    plt.show()

# Tets : 
image, mask = dataset[2]
display_image_with_mask(image, mask)


#%% Display image and mask overlayed

def display_image_mask_overlayed(image, mask, alpha=0.4):
    # Convert image and mask tensors to numpy arrays
    image = image.detach().numpy()
    mask = mask.detach().numpy()
    
    # Create a mask with color transparency
    color_mask = np.zeros_like(image)
    color_mask[..., 0] = mask * 255
    # Overlay the mask on the image
    overlay = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, color_mask.astype(np.uint8), alpha, 0)
    
    # Create a figure and axes
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis('off')
    plt.show()

# Tests : 
image, mask = dataset[2]
display_image_mask_overlayed(image, mask)