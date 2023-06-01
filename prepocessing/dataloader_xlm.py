#%% 

import os
from PIL import Image
import torch
import torchvision.transforms as T
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        annotations.append([xmin, ymin, xmax, ymax])
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
        annotation = torch.tensor(self.annotations[index])
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, annotation
    
    def __len__(self):
        return len(self.images)

#%% Tests : 
image_dir = "C:/Users/lucas.degeorge/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
annotation_dir = "C:/Users/lucas.degeorge/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations"

dataset = PascalVOCDataset(image_dir, annotation_dir, transform=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

#%% Display images 

def display_image_with_annotations(image, annotations):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    # Iterate over the annotations and draw bounding boxes
    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation
        bbox = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(bbox)
    plt.show()

# Tests 
image, annotations = dataset[14563]
display_image_with_annotations(image, annotations)