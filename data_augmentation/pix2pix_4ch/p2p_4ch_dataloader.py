#%% 
import os
import torch
from torchvision import transforms
from PIL import Image
import sys 

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation")
from preprocessing.size_reduction import resize

## Resize the labeled images 
image_path = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
mask_path = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"
output_path_images = "C:/Users/lucas.degeorge/Documents/Images/resized_images/resized_labeled_images"
output_path_masks = "C:/Users/lucas.degeorge/Documents/Images/resized_images/resized_masks"
where_write = "C:/Users/lucas.degeorge/Documents/Images/resized_images"

# resize(image_path, mask_path, output_path_A, output_path_B)

#%% 

## Combine images and masks
def combine_image_mask(image_folder, mask_folder, one_batch=False):
    image_files = os.listdir(image_folder)

    combined_tensors = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        mask_file = image_file.replace(".png", "_mask.png")
        mask_path = os.path.join(mask_folder, mask_file)

        image = Image.open(image_path).convert("L")
        image = transforms.ToTensor()(image)
        mask = Image.open(mask_path).convert("L")
        mask = transforms.ToTensor()(mask)

        if one_batch:
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)

            # Concatenate image and mask tensors
            combined_tensor = torch.cat((image, mask), dim=1)
            combined_tensors.append(combined_tensor)
        
        else:
            combined_tensor = torch.cat((image, mask), dim=0)
            combined_tensors.append(combined_tensor)

    if one_batch: return torch.cat(combined_tensors, dim=0) # Stack all combined tensors along the batch dimension
    else: return combined_tensors


# separate image and mask
def separate_image_mask(combined_tensor, return_PIL=False):
    # Split the combined tensor into image and mask tensors
    image = combined_tensor[0:1, :, :]
    mask = combined_tensor[1:2, :, :]

    image = image.squeeze(1)
    mask = mask.squeeze(1)

    if return_PIL:
        image = transforms.ToPILImage()(image)
        mask = transforms.ToPILImage()(mask)

    return image, mask


# tests : 
# combined_tensors = combine_image_mask(output_path_images, output_path_masks, one_batch=False)
# image, mask = separate_image_mask(combined_tensors[0], True)

def data_processing(image_folder, mask_folder, output_A, output_B, where_write):
    """ For all the images of image_folder: resize, combine with mask and save as a .pt file """
    resize(image_folder, mask_folder, output_A, output_B)
    combined_tensors = combine_image_mask(output_A, output_B, False)
    torch.save(combined_tensors, where_write + "/combined_data")

# data_processing(image_path, mask_path, output_path_images, output_path_masks, where_write)


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path, transform=None):
        self.transform = transform
        try:
            self.combined_images = torch.load(pt_path)
        except FileNotFoundError:
            print("file not found: ", pt_path)
            raise FileNotFoundError
    
    def __getitem__(self, index):
        image = self.combined_images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.combined_images)


def get_dataloader(pt_file, batch_size, shuffle=True, pin_memory=True):
    transform = None #transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CombinedDataset(pt_file, transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

