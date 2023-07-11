#%%
import torch
import json
from PIL import Image
import torchvision.transforms as T
import io
import matplotlib.pyplot as plt 

from generator import * 

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

model_folder = "C:/Users/lucas.degeorge/Documents/trained_models/pix2pix"
mask_folder = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"

#%% 

def generate(model_path, mask_path, class_values=[0,127,255], display=True, return_input=False):
    """
        model_path (string) : path to the model to use
        mask_path (string) : path to the mask
    """
    # load the model
    generator = UnetGenerator()
    with open(model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        generator.load_state_dict(torch.load(buffer), strict=False)
    generator.eval()

    # load the image
    mask = Image.open(mask_path).convert('L')
    if mask.size != (1024,1024): raise ValueError("Up to now, images can only have (1024,1024) shape")
    mask = T.functional.to_tensor(mask) * 255

    print(mask.shape)

    # generate
    prediction = generator(mask.unsqueeze(0)).squeeze()
    print(prediction.shape)
    # prediction = prediction.permute(1,2,0)
    # prediction = torch.softmax(prediction, dim=-1)
    # prediction = torch.argmax(prediction, dim=-1)
    # # prediction = torch.tensor(class_values, device=device)[prediction]

    if display:
        plt.subplot(1, 2, 1)
        plt.imshow(mask.permute(1,2,0).numpy())
        plt.title('Mask')
        plt.axis('off')
        # Display the mask
        plt.subplot(1, 2, 2)
        plt.imshow(prediction.detach().numpy()) #, cmap='gray')
        plt.title('Image generated')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    if return_input:
        return mask, prediction
    else:
        return prediction
    
#%% Tests

mask_test = mask_folder + "/0000001_mask.png"
model_test = model_folder + "/pix2pix_generator_20230703_112923.pth"

mask, pred = generate(model_test, mask_test, display=True, return_input=True)

