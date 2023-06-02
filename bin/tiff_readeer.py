#%% 
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import copy
from tifffile import imread
from aicsimageio import AICSImage

#%% Tifffile 

image_stack = imread("C:/Users/lucas.degeorge/Documents/data_predictions_apeer/000001_finalprediction.ome.tiff")
# image_stack = imread("C:/Users/lucas.degeorge/Downloads/annotation_test - export/000001_test_background.ome.tiff")
print(image_stack.shape)
print(image_stack.dtype)

# for i in range(3):
#     plt.imshow(image_stack[i])
#     plt.colorbar()
#     plt.show()

plt.imshow(image_stack[0])

histogram, bins = np.histogram(image_stack.flatten(), bins=256, range=[0, 256])
plt.figure(figsize=(10, 5))
plt.bar(bins[:-1], histogram, width=1)
plt.title("Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

#%% aicsimageio

# img = AICSImage("C:/Users/lucas.degeorge/Documents/data_predictions_apeer/000001_finalprediction.ome.tiff")
img = AICSImage("C:/Users/lucas.degeorge/Downloads/annotation_test - export/000001_test_background.ome.tiff")
img.data  # returns 6D STCZYX numpy array
img.dims  # returns string "STCZYX"
img.shape  # returns tuple of dimension sizes in STCZYX order

a = img.get_image_data("CZYX", S=0, T=0)[0,0,:,:]

plt.imshow(a)
plt.colorbar()
plt.show()

histogram, bins = np.histogram(a.flatten(), bins=256, range=[0, 256])
plt.figure(figsize=(10, 5))
plt.bar(bins[:-1], histogram, width=1)
plt.title("Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

#%% apeer_ometiff_library

from apeer_ometiff_library import io
import matplotlib.pyplot as plt

# Read the OME-TIFF file
file_path = "C:/Users/lucas/Documents/GitHub/Nanowire_image_segmentation/images_test/2_010_Nwire.ome.tiff"
ome_tiff = io.read_ometiff(file_path)


#%% histo

# Compute the histogram
histogram, bins = np.histogram(ome_tiff[0][0][0][0].flatten(), bins=256, range=[0, 256])

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(bins[:-1], histogram, width=1)
plt.title("Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.imshow(ome_tiff[0][0][0][0], cmap='gray')
plt.show()

