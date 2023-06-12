#%%
import cv2
import numpy as np
from PIL import Image
from apeer_ometiff_library import io
import os 
import matplotlib.pyplot as plt
import shutil

c2n = False

if c2n:
    input_annotation_folder = "C:/Users/lucas.degeorge/Documents/Images/annotations_renamed"
    # png_path = "C:/Users/lucas.degeorge/Documents/Images/apeer_png"
    outputA_folder = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
    mask_folder = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"

else:
    input_annotation_folder = "C:/Users/lucas/Desktop/annotations_renamed"
    input_images_folder = "C:/Users/lucas/Desktop/labeled_images_not_reshaped"
    output_images_folder = "C:/Users/lucas/Desktop/labeled_images"
    mask_folder = "C:/Users/lucas/Desktop/binary_masks"



#%% 

def get_mask(filename, folder_path, output_folder_path=mask_folder):
    reshaped = False

    image = Image.open(input_images_folder + "/" + filename + ".png")
    image = np.array(image)
    nanowire = io.read_ometiff(folder_path + "/" + filename + "_Nwire.ome.tiff")[0][0][0][0]
    droplet = io.read_ometiff(folder_path + "/" + filename + "_Ldrop.ome.tiff")[0][0][0][0]

    # reshape if necessary
    if nanowire.shape[0] > 1024:
        reshaped = True 
        nanowire = nanowire[:1024,:]
        image = image[:1024,:]
        droplet = droplet[:1024,:]
    if nanowire.shape[1] > 1024: 
        reshaped = True
        nanowire = nanowire[:,:1024]
        image = image[:,:1024]
        droplet = droplet[:,:1024]
    if droplet.shape[0] > 1024: 
        reshaped = True
        nanowire = nanowire[:1024,:]
        image = image[:1024,:]
        droplet = droplet[:1024,:]
    if droplet.shape[1] > 1024: 
        reshaped = True
        nanowire = nanowire[:,:1024]
        image = image[:,:1024]
        droplet = droplet[:,:1024]

    NW_intensity = 127  
    droplet_intensity = 255 
    mask = np.zeros_like(nanowire)
    mask[nanowire != 0] = NW_intensity
    mask[droplet != 0] = droplet_intensity

    # write the mask in the folder binary_masks
    cv2.imwrite(output_folder_path + "/" + filename + "_mask.png", mask.astype(np.uint8))

    # write the corresponding png image file in the folder labeled_images
    if reshaped:
        cv2.imwrite(output_images_folder + "/" + filename + ".png", image.astype(np.uint8))
    else:
        source_path = os.path.join(input_images_folder, f"{filename}.png")
        destination_path = os.path.join(output_images_folder, f"{filename}.png")
        shutil.copy2(source_path, destination_path)


for filename in os.listdir(input_annotation_folder):
    if filename.endswith("_Ldrop.ome.tiff"):

        x = filename.split("_Ldrop.ome.tiff")[0]  # Extract x from the filename
        ldrop_file = os.path.join(input_annotation_folder, filename)
        nwire_file = os.path.join(input_annotation_folder, filename.replace("_Ldrop", "_Nwire"))

        # Check if the corresponding Nwire file or the corresponding png image exist
        if os.path.isfile(nwire_file): 
            get_mask(x, input_annotation_folder)
        else:
            print(f"Corresponding Nwire file not found for x: {x}")
