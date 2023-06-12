#%% 
import sys
import json
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation") 

from preprocessing.dataloader import *
from losses import * 

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    trainer_arguments = arguments["trainer"]

#%% 

class Trainer:
    def __init__(self, model, arguments=trainer_arguments, ):
        self.model = model

        # supervised loss
        self.sup_loss_mode = arguments["sup_loss"]
        
        # unsupervised loss
        self.unsup_loss_mode = arguments["unsup_loss"]
         
        # optimizer 
        if arguments["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=arguments["optimizer_args"]["lr"], momentum=arguments["optimizer_args"]["momentum"])
        else:
            raise ValueError("optimizer has an invalid value. Must be in ['sgd']")
        
        self.nb_epochs = arguments["nb_epochs"]

        

