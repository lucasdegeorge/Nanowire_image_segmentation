#%%
import sys
import time
import torch
import json
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt 
import io

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/model") 
from dataloader import *
from model import * 
from preprocessing.display import *

model_folder = "C:/Users/lucas.degeorge/Documents/trained_models"
image_folder = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"

#%% 

def predict(model_path, image, class_values=[0,127,255], display=True, return_input=False):
    """
        model_path (string) : path to the model to use
        image (string) : path to the image
    """
    # load the model
    model = Model(mode="semi")
    with open(model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        model.load_state_dict(torch.load(buffer), strict=False)
    model.eval()

    # load the image
    converter = T.ToTensor()
    image = Image.open(image).convert('L')
    if image.size != (1024,1024): raise ValueError("Up to now, images can only have (1024,1024) shape")
    image = converter(image).to(device)

    # predict
    prediction = model(image.unsqueeze(0), eval=True)["output_l"][0]
    prediction = prediction.permute(1,2,0)
    prediction = torch.softmax(prediction, dim=-1)
    prediction = torch.argmax(prediction, dim=-1)
    # prediction = torch.tensor(class_values, device=device)[prediction]

    if display:
        display_image_with_mask(image, prediction)
        display_image_mask_overlayed(image, prediction)

    if return_input:
        return image, prediction
    else:
        return prediction

#%% Tests

image_test = image_folder + "/0000158.png"
model_test = model_folder + "/model_semi_20230616_111708_best.pth"

image, prediction = predict(model_test, image_test, display=True, return_input=True)


#%% Accuracy

def mIoU(output, target, nb_classes=3):
    """ adapted from: https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/utils/score.py
        output is a 4D tensor, target a 3D tensor
    """

    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1
    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()

    area_inter = torch.histc(intersection, bins=nb_classes, min=1, max=nb_classes)
    area_pred = torch.histc(predict, bins=nb_classes, min=1, max=nb_classes)
    area_lab = torch.histc(target, bins=nb_classes, min=1, max=nb_classes)
    area_union = area_pred + area_lab - area_inter

    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()

# tests 



