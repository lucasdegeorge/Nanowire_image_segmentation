#%%
import cv2
import numpy as np
from apeer_ometiff_library import io
import os 
import matplotlib.pyplot as plt

#%% 

def get_mask(filename, folder_path, output_folder_path='binary_masks_test/'):
    # reading file 
    nanowire = io.read_ometiff(folder_path + "/" + filename + "_Nwire.ome.tiff")[0][0][0][0]
    droplet = io.read_ometiff(folder_path + "/" + filename + "_Ldrop.ome.tiff")[0][0][0][0]

    NW_intensity = 127  
    droplet_intensity = 255 
    mask = np.zeros_like(nanowire)

    mask[nanowire != 0] = NW_intensity
    mask[droplet != 0] = droplet_intensity

    cv2.imwrite(output_folder_path + "/" + filename + "_mask.png", mask.astype(np.uint8))


#%% 

folder_path = "C:/Users/lucas/Documents/GitHub/Nanowire_image_segmentation/images_test"

for filename in os.listdir(folder_path):
    if filename.endswith("_Ldrop.ome.tiff"):

        x = filename.split("_Ldrop.ome.tiff")[0]  # Extract x from the filename
        ldrop_file = os.path.join(folder_path, filename)
        nwire_file = os.path.join(folder_path, filename.replace("_Ldrop", "_Nwire"))

        # Check if the corresponding Nwire file exists
        if os.path.isfile(nwire_file): 
            get_mask(x, folder_path)
        else:
            print(f"Corresponding Nwire file not found for x: {x}")
