#%% 
import os
import glob
import shutil

#%% 

def get_next_file_number(folder_path):
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    highest_number = 0

    # Find the highest existing numbering
    for file_name in png_files:
        file_number = int(os.path.splitext(file_name)[0])
        highest_number = max(highest_number, file_number)

    return highest_number + 1

def rename_files(folder_path, count_restart=False, counter_starter=None):
    if not(count_restart): next_number = get_next_file_number(folder_path)
    elif counter_starter is not None: next_number = counter_starter
    else: next_number = 1 

    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    png_files.sort() # Sort the PNG files alphabetically

    for file_name in png_files:
        new_file_name = f'{next_number:07}.png'
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)
        next_number += 1

def rename_files_two_folders(folderA_path, outputA_folder, folderB_path, outputB_folder, counter_start=1):
    png_files = glob.glob(os.path.join(folderA_path, '*.png'))
    png_files.sort()

    counter = counter_start
    for png_file in png_files:
        tiff_file_1 = os.path.join(folderB_path, os.path.splitext(os.path.basename(png_file))[0] + '_Ldrop.ome.tiff')
        tiff_file_2 = os.path.join(folderB_path, os.path.splitext(os.path.basename(png_file))[0] + '_Nwire.ome.tiff')

        if os.path.exists(tiff_file_1) and os.path.exists(tiff_file_2):
            new_png_name = '{:07d}.png'.format(counter)
            new_png_path = os.path.join(outputA_folder, new_png_name)
            new_tiff_name_1 = '{:07d}_Ldrop.ome.tiff'.format(counter)
            new_tiff_name_2 = '{:07d}_Nwire.ome.tiff'.format(counter)
            new_tiff_path_1 = os.path.join(outputB_folder, new_tiff_name_1)
            new_tiff_path_2 = os.path.join(outputB_folder, new_tiff_name_2)

            shutil.copy2(png_file, new_png_path)
            shutil.copy2(tiff_file_1, new_tiff_path_1)
            shutil.copy2(tiff_file_2, new_tiff_path_2)

            counter+=1
    return counter

png_path = "C:/Users/lucas.degeorge/Documents/Images/apeer_png_raw"
annotation_path = "C:/Users/lucas.degeorge/Documents/Images/apeer_annotations_raw"
output_png = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
output_anno = "C:/Users/lucas.degeorge/Documents/Images/annotations_renamed"
unlabeled_folder = "C:/Users/lucas.degeorge/Documents/Images/unlabeled_images"

# rename the labeled images from apeer
counter = rename_files_two_folders(png_path, output_png, annotation_path, output_anno)

# renamed all the unlabeled images
rename_files(unlabeled_folder, count_restart=True, counter_starter=counter)

#%% 

main_folder_path = "D:/Images_nanomax/unlabeled_images - Copie"

# Initialize a counter
count = 1

# Iterate over the subfolders and rename the PNG files
# for folder_name in os.listdir(main_folder_path):
#     folder_path = os.path.join(main_folder_path, folder_name)
#     if os.path.isdir(folder_path):

file_list = os.listdir(main_folder_path)
file_list.sort()  # Sort the file list if necessary

# Rename the files in the folder
for filename in file_list:
    if filename.endswith('.png'):
        new_filename = f'1_{count:06}.png'  # Format the new filename
        file_path = os.path.join(main_folder_path, filename)
        new_file_path = os.path.join(main_folder_path, new_filename)  # Save in main folder
        os.rename(file_path, new_file_path)
        count += 1