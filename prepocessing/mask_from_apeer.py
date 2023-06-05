#%%
import cv2
import numpy as np
from apeer_ometiff_library import io
import os 
import matplotlib.pyplot as plt
import shutil

input_folder = "C:/Users/lucas.degeorge/Documents/Images/apeer_annotations"
png_path = "C:/Users/lucas.degeorge/Documents/Images/apeer_png"
outputA_folder = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
mask_folder = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"

#%% 

def get_mask(filename, folder_path, output_folder_path=mask_folder):
    nanowire = io.read_ometiff(folder_path + "/" + filename + "_Nwire.ome.tiff")[0][0][0][0]
    droplet = io.read_ometiff(folder_path + "/" + filename + "_Ldrop.ome.tiff")[0][0][0][0]

    NW_intensity = 127  
    droplet_intensity = 255 
    mask = np.zeros_like(nanowire)
    mask[nanowire != 0] = NW_intensity
    mask[droplet != 0] = droplet_intensity

    # write the mask in the folder binary_masks
    cv2.imwrite(output_folder_path + "/" + filename + "_mask.png", mask.astype(np.uint8))

    # write the corresponding png file in the folder labeled_images
    source_path = os.path.join(png_path, f"{filename}.png")
    destination_path = os.path.join(outputA_folder, f"{filename}.png")
    shutil.copy2(source_path, destination_path)


for filename in os.listdir(input_folder):
    if filename.endswith("_Ldrop.ome.tiff"):

        x = filename.split("_Ldrop.ome.tiff")[0]  # Extract x from the filename
        ldrop_file = os.path.join(input_folder, filename)
        nwire_file = os.path.join(input_folder, filename.replace("_Ldrop", "_Nwire"))
        png_file = os.path.join()

        # Check if the corresponding Nwire file or the corresponding png image exist
        if os.path.isfile(nwire_file): 
            get_mask(x, input_folder)
        else:
            print(f"Corresponding Nwire file not found for x: {x}")
