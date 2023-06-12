#%% 
import os
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt 
import numpy as np

batch_size = 32 

c2n = True

if c2n:
    labeled_image_dir = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
    masks_dir = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"
    unlabeled_image_dir = "C:/Users/lucas.degeorge/Documents/Images/unlabeled_images"
    folder_where_write = "C:/Users/lucas.degeorge/Documents/Images"
else:
    labeled_image_dir = "C:/Users/lucas/Desktop/labeled_images"
    masks_dir = "C:/Users/lucas/Desktop/binary_masks"
    unlabeled_image_dir = "C:/Users/lucas/Desktop/unlabeled_images"
    folder_where_write = "C:/Users/lucas/Desktop"

filetype = '.png'

#%% mask converter 

def mask_converter(mask, out="one-hot", nb_classes=3, class_values=[0,127,255]):
    """ mask can be:
        - a string representing the path to a filetype image -> the mask converted in a (nb_classes, H, W)  tensor(s)
        - a (.,H,W,1) tensor -> the mask is converted as a (.,nb_classes,H,W) tensor 
        - a (.,nb_classes,H,W) tensor -> the mask is converted as a (.,H,W,1) tensor
        (tensors can represent a batch)
    """
    if type(mask) == str:  # mask is a path, the mask is converted in a tensor 
        image = Image.open(mask).convert("L")
        # if image.size != (1024,1024): image = image.resize((1024,1024))
        tensor_image = T.functional.to_tensor(image) * 255
        tensor_image = tensor_image.to(torch.uint8) 
        # if tensor_image.size != (1024,1024): tensor_image = tensor_image.resize((1024,1024))
        for i in range(nb_classes):
            tensor_image = torch.where(tensor_image == class_values[i], torch.tensor(i), tensor_image)         
        try:
            tensor_image = torch.nn.functional.one_hot(tensor_image.to(torch.int64), nb_classes).permute(0,3,1,2).squeeze(0)
        except RuntimeError:
            raise RuntimeError("Error while trying to convert the file" + mask)
        if out == "one-hot":
            return tensor_image
        elif out == "image-like":
            return tensor_image.permute(1,2,0)
        else: 
            raise ValueError("out mode is incorrect")
    if type(mask) == torch.Tensor:
        if out == "one-hot":
            assert mask.shape[-1] == nb_classes, "Tensor has a wrong shape"
            if len(mask.shape) == 3: return mask.permute(2,0,1)
            elif len(mask.shape) == 4: return mask.permute(0,3,1,2)
            else: raise ValueError("Tensor has a wrong shape")
        if out == "image-like":
            assert mask.shape[0] == nb_classes or mask.shape[1] == nb_classes, "Tensor has a wrong shape"
            if len(mask.shape) == 3: return mask.permute(1,2,0)
            elif len(mask.shape) == 4: return mask.permute(0,2,3,1)
            else: raise ValueError("Tensor has a wrong shape")



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
            if image.size != (1024,1024): image = image.resize((1024,1024))
            image = converter(image)
            images.append(image)
            if mask_folder is not None:
                mask_path = os.path.join(mask_folder, filename[:-4] + '_mask.png')
                if os.path.isfile(mask_path):
                    try:
                        mask = mask_converter(mask_path)
                        masks.append(mask)
                    except RuntimeError:
                        print("mask" + mask_path + "has not been saved")
                        pass
    # save tensors 
    file_name = folder_where_write + "/" + image_folder.split("/")[-1] + ".pt"
    torch.save(images, file_name)
    if mask_folder is not None:
        folder_name = folder_where_write + "/" + mask_folder.split("/")[-1] + ".pt"
        torch.save(masks, folder_name)

# save_and_load(labeled_image_dir, masks_dir)
# save_and_load(unlabeled_image_dir, None)
# labeled_images = torch.load(folder_where_write + "/" + "labeled_images.pt")
# masks = torch.load(folder_where_write + "/" + "binary_masks.pt")
# unlabeled_images = torch.load(folder_where_write + "/" + "unlabeled_images.pt")

#%% dataset and dataloader

# Labeled data and masks

def load_labeled_data(image_dir, annotation_dir):
    try:
        labeled_images = torch.load(folder_where_write + "/" + "labeled_images.pt")
        masks = torch.load(folder_where_write + "/" + "binary_masks.pt")
    except FileNotFoundError:
        print("files labeled_images.pt and binary_masks.pt not found")
        save_and_load(image_dir, annotation_dir)
        labeled_images = torch.load(folder_where_write + "/" + "labeled_images.pt")
        masks = torch.load(folder_where_write + "/" + "binary_masks.pt")
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
        unlabeled_images = torch.load(folder_where_write + "/" + "unlabeled_images.pt")
    except FileNotFoundError:
        print("file unlabeled_images.pt not found")
        save_and_load(image_dir, None)
        unlabeled_images = torch.load(folder_where_write + "/" + "unlabeled_images.pt")
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
print("dataloaders ok")


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
