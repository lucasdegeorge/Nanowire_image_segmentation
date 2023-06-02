import os

def get_next_file_number(folder_path):
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    highest_number = 0

    # Find the highest existing numbering
    for file_name in png_files:
        file_number = int(os.path.splitext(file_name)[0])
        highest_number = max(highest_number, file_number)

    return highest_number + 1

def rename_files(folder_path, count_restart=False):
    if count_restart: next_number = get_next_file_number(folder_path)
    else: next_number = 1

    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    png_files.sort() # Sort the PNG files alphabetically

    for file_name in png_files:
        new_file_name = f'{next_number:06}.png'
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)
        # print(f'Renamed {file_name} to {new_file_name}')
        next_number += 1


folder_path = 'path/to/your/folder'
rename_files(folder_path)