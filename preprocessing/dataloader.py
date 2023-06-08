#%% 

import os
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt 

batch_size = 32 

labeled_image_dir = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
masks_dir = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"
unlabeled_image_dir = "C:/Users/lucas.degeorge/Documents/Images/unlabeled_images"

filetype = '.png'

#%% save tensors in .pt 

def save_and_load(image_folder, mask_folder=None):
    """ loads the images in the folders and save save in a .pt file using the same name"""
    converter = T.ToTensor()
    images = []
    masks = []
    # load and converter images 
    for filename in os.listdir(image_folder):
        if filename.endswith(filetype):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('L')
            # if image.size != (1024,1024): image = image.resize((1024,1024))
            image = converter(image)
            images.append(image)
            if mask_folder is not None:
                mask_path = os.path.join(mask_folder, filename[:-4] + '_mask.png')
                if os.path.isfile(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    # if mask.size != (1024,1024): mask = mask.resize((1024,1024))
                    mask = converter(mask)
                    masks.append(mask)
    # save tensors 
    folder_name = image_folder.split("/")[-1] + ".pt"
    torch.save(images, folder_name)
    if mask_folder is not None:
        folder_name = mask_folder.split("/")[-1] + ".pt"
        torch.save(masks, folder_name)

save_and_load(labeled_image_dir, masks_dir)
save_and_load(unlabeled_image_dir, None)
labeled_images = torch.load("labeled_images.pt")
masks = torch.load("binary_masks.pt")
unlabeled_images = torch.load("unlabeled_images.pt")

#%% dataset and dataloader

# Labeled data and masks

def load_labeled_data(image_dir, annotation_dir):
    try:
        labeled_images = torch.load("labeled_images.pt")
        masks = torch.load("binary_masks.pt")
    except FileNotFoundError:
        save_and_load(image_dir, annotation_dir)
        labeled_images = torch.load("labeled_images.pt")
        masks = torch.load("binary_masks.pt")
    return labeled_images, masks


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images, self.masks = load_labeled_data(image_dir, annotation_dir)
    
    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
    
    def __len__(self):
        return len(self.images)


# Unlabeled data 

def load_unlabeled_data(image_dir):
    try:
        unlabeled_images = torch.load("unlabeled_images.pt")
    except FileNotFoundError:
        print("file unlabeled_images.pt not found")
        save_and_load(image_dir, None)
        unlabeled_images = torch.load("unlabeled_images.pt")
    return unlabeled_images


class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = load_unlabeled_data(image_dir)
    
    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.images)



# Definition 

labeled_dataset = LabeledDataset(labeled_image_dir, masks_dir, transform=None)
unlabeled_dataset = UnlabeledDataset(unlabeled_image_dir, transform=None)

labeled_dataloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

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
# image, mask = labeled_dataset[210]
# display_image_with_mask(image, mask)


#%% Display image and mask overlayed

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
# display_image_mask_overlayed(image, mask)
