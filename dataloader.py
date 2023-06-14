#%% 
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
            tensor_image = tensor_image.to(torch.float32)
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

def save_and_load(image_folder, mask_folder=None, folder_where_write=folder_where_write):
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

def load_labeled_data(image_dir, annotation_dir, folder_where_write):
    try:
        labeled_images = torch.load(folder_where_write + "/" + "labeled_images.pt")
        masks = torch.load(folder_where_write + "/" + "binary_masks.pt")
    except FileNotFoundError:
        print("files labeled_images.pt and binary_masks.pt not found")
        save_and_load(image_dir, annotation_dir, folder_where_write)
        labeled_images = torch.load(folder_where_write + "/" + "labeled_images.pt")
        masks = torch.load(folder_where_write + "/" + "binary_masks.pt")
    train_images, eval_images, train_masks, eval_masks = train_test_split(labeled_images, masks, test_size=0.2, random_state=42)
    return train_images, eval_images, train_masks, eval_masks


class train_LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, transform=None):
        self.transform = transform
        self.images = images
        self.masks = masks
        # self.images = [ t.to(device) for t in images]
        # self.masks = [ t.to(device) for t in masks]
    
    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
    
    def __len__(self):
        return len(self.images)
    
class eval_LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, transform=None):
        self.transform = transform
        self.images = images
        self.masks = masks
        # self.images = [ t.to(device) for t in images]
        # self.masks = [ t.to(device) for t in masks]
    
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

def load_unlabeled_data(image_dir, folder_where_write):
    try:
        unlabeled_images = torch.load(folder_where_write + "/" + "unlabeled_images.pt")
    except FileNotFoundError:
        print("file unlabeled_images.pt not found")
        save_and_load(image_dir, None, folder_where_write)
        unlabeled_images = torch.load(folder_where_write + "/" + "unlabeled_images.pt")
    # unlabeled_images = [ t.to(device) for t in unlabeled_images]
    return unlabeled_images


class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None, folder_where_write=folder_where_write):
        self.image_dir = image_dir
        self.transform = transform
        self.images = load_unlabeled_data(image_dir, folder_where_write)
    
    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.images)


# Definition 

def get_dataloaders(batch_size, labeled_image_dir=labeled_image_dir, masks_dir=masks_dir, unlabeled_image_dir=unlabeled_image_dir, folder_where_write=folder_where_write):
    train_images, eval_images, train_masks, eval_masks = load_labeled_data(labeled_image_dir, masks_dir, folder_where_write=folder_where_write)

    train_labeled_dataset = train_LabeledDataset(train_images, train_masks, transform=None)
    eval_labeled_dataset = eval_LabeledDataset(eval_images, eval_masks, transform=None)
    unlabeled_dataset = UnlabeledDataset(unlabeled_image_dir, transform=None)

    train_labeled_dataloader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    eval_labeled_dataloader = torch.utils.data.DataLoader(eval_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    return train_labeled_dataloader, eval_labeled_dataloader,  unlabeled_dataloader

# train_labeled_dataloader, eval_labeled_dataloader,  unlabeled_dataloader = get_dataloaders(batch_size=batch_size)
