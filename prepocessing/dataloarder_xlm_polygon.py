#%% 

import os
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

#%% 

def load_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Parse segmentation mask (polygon vertices)
        polygon = obj.find('polygon')
        if polygon is not None:
            vertices = []
            for point in polygon.findall('pt'):
                x = float(point.find('x').text)
                y = float(point.find('y').text)
                vertices.append((x, y))
            annotations.append({
                'bbox': [xmin, ymin, xmax, ymax],
                'segmentation': vertices
            })
        else:
            annotations.append({
                'bbox': [xmin, ymin, xmax, ymax]
            })
    return annotations


def load_data(image_dir, annotation_dir):
    images = []
    annotations = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            xml_path = os.path.join(annotation_dir, filename[:-4] + '.xml')
            if os.path.isfile(xml_path):
                images.append(image_path)
                annotations.append(load_annotations(xml_path))
    return images, annotations

class PascalVOCDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images, self.annotations = load_data(image_dir, annotation_dir)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        annotation = self.annotations[index]
        mask = self.create_mask(annotation['segmentation'], image.size)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, mask
    
    def __len__(self):
        return len(self.images)
    
    def create_mask(self, segmentation, image_size):
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(segmentation, fill=1)
        return torch.tensor(np.array(mask), dtype=torch.float32)


#%% Tests : 
image_dir = "C:/Users/lucas.degeorge/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
annotation_dir = "C:/Users/lucas.degeorge/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations"

dataset = PascalVOCDataset(image_dir, annotation_dir, transform=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


