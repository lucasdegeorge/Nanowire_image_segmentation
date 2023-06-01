#%%
import json
import os
import numpy as np
import PIL.Image
import cv2

with open("annotations.json", "r") as read_file:
    data = json.load(read_file)

all_file_names=list(data.keys())

Files_in_directory = []
for root, dirs, files in os.walk("sample_frames"):
    for filename in files:
        Files_in_directory.append(filename)
        
for j in range(len(all_file_names)): # for each file
    image_name=data[all_file_names[j]]['filename']
    if image_name in Files_in_directory: 
         img = np.asarray(PIL.Image.open('sample_frames/'+image_name))    
    else:
        continue
    
    if data[all_file_names[j]]['regions'] != {}:
        #cv2.imwrite('images/%05.0f' % j +'.jpg',img)
        # print(j)
        shape_wire_x = data[all_file_names[j]]['regions'][0]['shape_attributes']['all_points_x']
        shape_wire_y = data[all_file_names[j]]['regions'][0]['shape_attributes']['all_points_y']
        shape_droplet_x = data[all_file_names[j]]['regions'][1]['shape_attributes']['all_points_x']
        shape_droplet_y = data[all_file_names[j]]['regions'][1]['shape_attributes']['all_points_y']

        ab_wire = np.stack((shape_wire_x, shape_wire_y), axis=1)
        ab_droplet = np.stack((shape_droplet_x, shape_droplet_y), axis=1)
        # img2 = cv2.drawContours(img, [ab], -1, (255,255,255), -1)
        mask = np.zeros((img.shape[0],img.shape[1]))
        img1 = cv2.drawContours(mask, [ab_wire], -1, 255, -1)
        img2 = cv2.drawContours(mask, [ab_droplet], -1, 127, -1)
        
        cv2.imwrite('binary_masks/%05.0f' % j +'.png',mask.astype(np.uint8))
        