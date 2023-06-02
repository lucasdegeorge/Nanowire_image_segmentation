
#%% 
import os
import zipfile
from skimage import io

# Specify the input folder containing the zip files
input_folder = "C:/Users/lucas.degeorge/Documents/Images/zip"

# Specify the output folder for the extracted and converted files
output_folder = "C:/Users/lucas.degeorge/Documents/Images/labeled_apeer"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.zip'):
        # Extract the zip file
        zip_path = os.path.join(input_folder, filename)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)

        # Loop through the extracted files and convert czi to png
        extracted_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
        for extracted_file in os.listdir(extracted_folder):
            if extracted_file.endswith('.czi'):
                czi_path = os.path.join(extracted_folder, extracted_file)
                output_file = os.path.splitext(extracted_file)[0] + '.png'
                output_path = os.path.join(output_folder, output_file)
                image = io.imread(czi_path)
                io.imsave(output_path, image)

        # Remove the extracted folder
        os.remove(zip_path)

print('Extraction and conversion completed.')