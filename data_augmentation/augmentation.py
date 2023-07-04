#%% 
import os
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import albumentations as A

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation")
from dataloader import mask_converter

labeled_image_dir = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
masks_dir = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"

#%% 

converter = T.ToTensor()
images = []
masks = []

for filename in os.listdir(labeled_image_dir):
    if filename.endswith("png"):
        image_path = os.path.join(labeled_image_dir, filename)
        image = Image.open(image_path).convert("L")
        # if image.size != (1024,1024): image = image.resize((1024,1024))
        # image = converter(image)
        images.append(image)
        if masks_dir is not None:
            mask_path = os.path.join(masks_dir, filename[:-4] + '_mask.png')
            if os.path.isfile(mask_path):
                try:
                    # mask = mask_converter(mask_path)
                    mask = Image.open(mask_path).convert("L")
                    # mask = converter(mask)
                    masks.append(mask)
                except RuntimeError:
                    print("mask" + mask_path + "has not been saved")
                    pass

transform = A.Compose([
    A.Rotate(limit=30, p=0.5),         
    A.HorizontalFlip(p=0.5),           
    A.VerticalFlip(p=0.5),             
    A.RandomBrightnessContrast(p=0.7)])

os.makedirs("C:/Users/lucas.degeorge/Documents/Images/augmented_dataset/images", exist_ok=True)
os.makedirs("C:/Users/lucas.degeorge/Documents/Images/augmented_dataset/masks", exist_ok=True)

augmented_images = []
augmented_masks = []

desired_dataset_size = 1000

while len(augmented_images) < desired_dataset_size:
    for image, mask in zip(images, masks):
        image_array = np.array(image)
        mask_array = np.array(mask)

        augmented = transform(image=image_array, mask=mask_array)
        augmented_image = augmented["image"]
        augmented_mask = augmented["mask"]

        augmented_image = Image.fromarray(augmented_image)
        augmented_mask = Image.fromarray(augmented_mask)


        augmented_image_path = os.path.join("C:/Users/lucas.degeorge/Documents/Images/augmented_dataset/images", "al_{}.png".format(226 + len(augmented_images)))
        augmented_mask_path = os.path.join("C:/Users/lucas.degeorge/Documents/Images/augmented_dataset/masks", "al_{}_mask.png".format(226 + len(augmented_masks)))

        augmented_image.save(augmented_image_path)
        augmented_mask.save(augmented_mask_path)

        augmented_images.append(augmented_image_path)
        augmented_masks.append(augmented_mask_path)

