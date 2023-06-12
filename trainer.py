#%% 
import sys
import json
import torch
from itertools import cycle
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
    def __init__(self, model, labeled_loader, unlabeled_loader, eval_loader, mode='semi', arguments=trainer_arguments):
        self.mode = mode
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
        self.running_loss = 0.
        self.last_loss = 0.

        # data loaders
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.eval_loader = eval_loader

    def train_one_epoch(self, epoch_idx):
        if self.mode == 'super':
            dataloader = iter(self.labeled_loader)
        if self.mode == 'semi':
            dataloader = iter(zip(cycle(self.labeled_loader), self.unlabeled_loader))

        self.model.train()

        





        

