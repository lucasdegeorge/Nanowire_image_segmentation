
#%% 
import os
import zipfile
from skimage import io
import czifile
import matplotlib.pyplot as plt

input_folder = "C:/Users/lucas.degeorge/Documents/Images/zip"
inter_folder = "C:/Users/lucas.degeorge/Documents/Images/labeled_apeer_inter"
output_folder = "C:/Users/lucas.degeorge/Documents/Images/labeled_apeer"

#%% Extract the files from zip files

os.makedirs(inter_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.zip'):
        zip_path = input_folder + "/" + filename
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(inter_folder)
        os.remove(zip_path)


#%% Convert czi files into png files

for filename in os.listdir(inter_folder):
    if filename.endswith('.czi'):
        czi_path = os.path.join(inter_folder, filename)
        output_file = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_folder, output_file)
        image = czifile.imread(czi_path)
        # plt.imshow(image[0,0,0,0,:,:,0])
        io.imsave(output_path, image[0,0,0,0,:,:,0])
