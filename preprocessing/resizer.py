#%% 
from PIL import Image
import os

folder_path = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path) and any(file_path.endswith(extension) for extension in ['.jpg', '.jpeg', '.png']):
        try:
            image = Image.open(file_path)
            width, height = image.size

            if width != 1024 or height != 1024:
                resized_image = image.resize((1024, 1024))
                resized_image.save(file_path)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")