#%% 
import os
import random
import shutil

def select_random_files(source_folder, destination_folder, num_files):
    # Get a list of PNG files in the source folder
    png_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]

    # Select a random sample of PNG files
    selected_files = random.sample(png_files, num_files)

    # Copy the selected files to the destination folder
    for file_name in selected_files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)

source_folder = "D:/Images_nanomax/Images/unlabeled_images_t1_60000"
destination_folder = "D:/Images_nanomax/Images/unlabeled_images_t1"
num_files = 40000

select_random_files(source_folder, destination_folder, num_files)