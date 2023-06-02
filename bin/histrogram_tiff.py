#%% 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the TIFF image
image_path = "C:/Users/lucas.degeorge/Downloads/annotation_test - export/000001_test_background.ome.tiff"
image = Image.open(image_path)

# Convert the image to grayscale
image_gray = image.convert("L")

# Convert the image to a NumPy array
image_array = np.array(image_gray)
plt.imshow(image_array)

# Compute the histogram
histogram, bins = np.histogram(image_array.flatten(), bins=256, range=[0, 256])

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(bins[:-1], histogram, width=1)
plt.title("Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()