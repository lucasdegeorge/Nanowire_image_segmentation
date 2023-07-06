#%% 
import cv2
import numpy as np
import os
from PIL import Image
import PIL


def resize(image_folder, mask_folder, output_folder_A, output_folder_B):
    images = []
    masks = []
    # load and converter images 
    for filename in os.listdir(image_folder):
        if filename.endswith("png"):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("L")
            image = image.resize((1024,1024), resample=PIL.Image.NEAREST)
            output_image_path = os.path.join(output_folder_A, filename)
            image.save(output_image_path)
            if mask_folder is not None:
                mask_path = os.path.join(mask_folder, filename[:-4] + '_mask.png')
                if os.path.isfile(mask_path):
                    mask = Image.open(mask_path).convert("L")
                    mask = mask.resize((1024,1024), resample=PIL.Image.NEAREST)
                    output_mask_path = os.path.join(output_folder_B, filename[:-4] + '_mask.png')
                    mask.save(output_mask_path)
    

image_path = "C:/Users/lucas.degeorge/Documents/Images/micro_batch_for_tests/resized_images"
mask_path = "C:/Users/lucas.degeorge/Documents/Images/micro_batch_for_tests/resized_masks"
output_path_A = "C:/Users/lucas.degeorge/Documents/Images/micro_batch_for_tests/enlarged_images"
output_path_B = "C:/Users/lucas.degeorge/Documents/Images/micro_batch_for_tests/enlarged_masks"

# resize(image_path, mask_path, output_path_A, output_path_B)
