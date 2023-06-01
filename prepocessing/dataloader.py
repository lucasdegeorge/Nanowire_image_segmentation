#%% 

import os
from PIL import Image
import torch
import torchvision.transforms as T

batch_size = 32 

labeled_image_dir = "C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/prepocessing/labeled_images"
annotation_dir = "C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/prepocessing/binary_masks"
unlabeled_image_dir = "C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/prepocessing/unlabeled_images"


#%% 

# Labeled data and masks

def load_labeled_data(image_dir, annotation_dir):
    labeled_images = []
    masks = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(annotation_dir, filename[:-4] + '.png')
            if os.path.isfile(mask_path):
                labeled_images.append(image_path)
                masks.append(mask_path)
    return labeled_images, masks

print(load_labeled_data(labeled_image_dir, annotation_dir)[0])


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
        if filename.endswith('.jpg'):
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

labeled_dataset = LabeledDataset(labeled_image_dir, annotation_dir, transform=None)
print(labeled_dataset.__len__())
unlabeled_dataset = UnlabeledDataset(unlabeled_image_dir, transform=None)
print(unlabeled_dataset.__len__())

labeled_dataloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=False)
unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)
