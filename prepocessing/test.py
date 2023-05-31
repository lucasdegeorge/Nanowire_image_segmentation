#%% 
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

#%% 

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
file_path = "C:/Users/lucas/Documents/GitHub/Nanowire_image_segmentation/000000_finalprediction.ome.tiff"
file_path_annot = "C:/Users/lucas/Documents/GitHub/Nanowire_image_segmentation/000001_test_background.ome.tiff"
ome_tiff = io.read_ometiff(file_path_annot)

# Access the individual images
images = ome_tiff[0]
print(images.shape)

for i in range(3):
    plt.figure()
    plt.imshow(images[0,0,i,:,:])
    plt.show()