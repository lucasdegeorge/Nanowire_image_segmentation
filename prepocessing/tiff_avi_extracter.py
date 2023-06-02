#%%
from PIL import Image
import cv2
import os

#%% Tiff extracter 

def extract_frames_from_tiff(tiff_file, output_folder, output_format='png'):
    tiff_image = Image.open(tiff_file)

    for i in range(tiff_image.n_frames): # for each frame 
        tiff_image.seek(i)
        rgb_image = tiff_image.convert('RGB')

        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f'frame13_{i}.{output_format}')
        rgb_image.save(output_file)
        # print(f'Saved frame {i} as {output_file}')
    tiff_image.close()


tiff_file_path = "C:/Users/lucas.degeorge/Documents/Images/tiff_to_convert/13_VIDEO.tif" # replace file name
output_folder = "C:/Users/lucas.degeorge/Documents/Images/unlabeled_images"
extract_frames_from_tiff(tiff_file_path, output_folder,output_format='png')

#%% AVI extracter 

def extract_frames_from_avi(avi_file, output_folder, output_format='png'):
    video = cv2.VideoCapture(avi_file)
    frame_count = 0
    while video.isOpened():
        # Read the next frame
        ret, frame = video.read()

        # Check if the frame was read successfully
        if ret:
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f'frame3.40_{frame_count}.{output_format}')
            cv2.imwrite(output_file, frame)
            # print(f'Saved frame {frame_count} as {output_file}')
            frame_count += 1
        else:
            break
    video.release()

avi_file_path = "C:/Users/lucas.degeorge/Documents/Images/videos_to_convert/3.40fps.avi" # replace file
output_folder = "C:/Users/lucas.degeorge/Documents/Images/unlabeled_images"
extract_frames_from_avi(avi_file_path, output_folder, output_format='png')