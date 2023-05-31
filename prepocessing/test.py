#%% 
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

img_test = mpimg.imread("C:/Users/lucas.degeorge/Documents/data_predictions_apeer/000001_finalprediction.ome.tiff")
img_test.shape
# plt.imshow(img_test) # [:,:,3])
# plt.colorbar()
# plt.show()

# a = img_test[:,:,3] != 255

# a.any()

#%% 

from apeer_ometiff_library import io
import matplotlib.pyplot as plt

# Read the OME-TIFF file
file_path = "path_to_your_ome.tif"
ome_tiff = io.read_ometiff(file_path)

# Access the individual images
images = ome_tiff["images"]

# Visualize each image
for image_index, image_data in enumerate(images):
    # Create a new figure and plot the image
    plt.figure()
    plt.imshow(image_data, cmap='gray')
    plt.title(f"Image {image_index+1}")

    # Display the plot
    plt.show()