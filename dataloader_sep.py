#%% 
import torch
import torchvision.transforms as transforms
import glob
from PIL import Image
import os
from sklearn.model_selection import train_test_split

labeled_image_dir = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
masks_dir = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"
unlabeled_image_dir = "D:/Images_nanomax/Images/unlabeled_images_t1" # "C:/Users/lucas.degeorge/Documents/Images/little_unlabeled_images" 
folder_where_write_A = "C:/Users/lucas.degeorge/Documents/Images"
folder_where_write_B = "D:/Images_nanomax/Images"

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
        tensor_image = transforms.functional.to_tensor(image) * 255
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

#%% 

converter = transforms.ToTensor()

def save_1by1(input_folder, output_folder, is_mask=False):

    i=0
    for file_path in glob.glob(input_folder + '/*.png'):
        if i % 1000 == 0: print(i)
        if is_mask:
            tensor = mask_converter(file_path)
        else:
            image = Image.open(file_path).convert("L")
            tensor = converter(image)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_folder, file_name + '.pt')
        torch.save(tensor, output_path)
        i+=1

# save_1by1(labeled_image_dir, folder_where_write_A + "/labeled_images_pt")
# save_1by1(masks_dir, folder_where_write_A + "/binary_masks_pt", is_mask=True)
# save_1by1(unlabeled_image_dir, folder_where_write_B + "/unlabeled_images_t1_pt")


#%% 

class train_LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, mask_list, transform=None):
        self.image_list = image_list
        self.mask_list = mask_list

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index]
        mask = self.mask_list[index]
        image = torch.load(image)
        mask = torch.load(mask)
        return image, mask

class eval_LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, mask_list, transform=None):
        self.image_list = image_list
        self.mask_list = mask_list

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index]
        mask = self.mask_list[index]
        image = torch.load(image)
        mask = torch.load(mask)
        return image, mask
    

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, transform=None):
        self.image_list = glob.glob(image_path + '/*.pt')

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index]
        image = torch.load(image)
        return image

def get_dataloaders_sep(batch_size, unlabeled=True, split=True, labeled_image_dir=labeled_image_dir + "_pt", masks_dir=masks_dir + "_pt", unlabeled_image_dir=unlabeled_image_dir + "_pt"):
    
    if split:
        image_list = glob.glob(labeled_image_dir + '/*.pt')
        mask_list = glob.glob(masks_dir + '/*.pt' )
        train_images, eval_images, train_masks, eval_masks = train_test_split(image_list, mask_list, test_size=0.2)
        train_dataset = train_LabeledDataset(train_images, train_masks)
        eval_dataset = eval_LabeledDataset(eval_images, eval_masks)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

        if unlabeled:
            unlabeled_dataset = UnlabeledDataset(unlabeled_image_dir)
            unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
            return train_dataloader, eval_dataloader, unlabeled_dataloader
        
        return train_dataloader, eval_dataloader


    else: 
        image_list = glob.glob(labeled_image_dir + '/*.pt')
        mask_list = glob.glob(masks_dir + '/*.pt' )
        train_dataset = train_LabeledDataset(image_list, mask_list)
        labeled_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
        
        if unlabeled:
            unlabeled_dataset = UnlabeledDataset(unlabeled_image_dir)
            unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
            return labeled_dataloader, unlabeled_dataloader
        
        return labeled_dataloader

# train_labeled_dataloader, eval_labeled_dataloader, unlabeled_dataloader = get_dataloaders_sep(batch_size=32, unlabeled=True)