import os
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import json

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    batch_size = arguments["batch_size"]
    device = arguments["device"]
    device = torch.device(device)
#%% One-hot to images 

def one_hot_to_image(mask, nb_classes=3, class_values=[0,127,255]):
    """ masks is a (H,W,nb_classes) tensor """
    if mask.shape[-1] != nb_classes:
        raise ValueError("mask.shape is incorrect")
    else:
        mask = torch.argmax(mask, dim=-1)
        mask = torch.tensor(class_values)[mask]
        return mask

#%% Display images and masks separated

def display_image_with_mask(image, mask):
    image = image.permute(1, 2, 0).numpy()
    mask = mask.squeeze().numpy()

    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')
    # Display the mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Test : 
# image, mask = labeled_dataset[200]
# mask = one_hot_to_image(mask_converter(mask, out="image-like"))
# display_image_with_mask(image, mask)


#%% Display image and mask overlayed ## May not work well since last update

def display_image_mask_overlayed(image, mask, alpha=0.2):
    image = image.permute(1, 2, 0).numpy()
    mask = mask.squeeze().numpy()

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    cmap = plt.get_cmap('jet')
    mask_colored = cmap(mask / mask.max()) 
    ax.imshow(mask_colored, alpha=alpha)
    plt.show()

# Tests : 
# image, mask = labeled_dataset[78]
# mask = one_hot_to_image(mask_converter(mask, out="image-like"))
# display_image_mask_overlayed(image, mask)