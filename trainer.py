#%% 
import sys
import json
import torch
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/model") 

from preprocessing.dataloader import *
from losses import * 
from model import * 

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
        self.weight_ul = arguments["weight_ul"]
         
        # optimizer 
        if arguments["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=arguments["optimizer_args"]["lr"], momentum=arguments["optimizer_args"]["momentum"])
        else:
            raise ValueError("optimizer has an invalid value. Must be in ['sgd']")
        
        self.nb_epochs = arguments["nb_epochs"]

        # data loaders
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.eval_loader = eval_loader

    def train_super_1epoch(self, epoch_idx, tb_writer):
        dataloader = iter(self.labeled_loader)
        self.model.train()

        running_loss = 0.
        last_loss = 0.

        for i, (x_l, target_l) in enumerate(dataloader):
            self.optimizer.zero_grad()
            output_l = self.model(x_l, None)["output_l"]

            loss = supervised_loss(output_l, target_l, mode=self.sup_loss_mode)
            loss.backward()
            self.optimizer.step()

            # report data
            running_loss += loss.item()
            if i % 10 == 0:
                last_loss = running_loss / 10
                print('  batch {} loss: {}'.format(i, last_loss))
                tb_x = epoch_idx * len(dataloader) + i
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    
    def train_semi_1epoch(self, epoch_idx, tb_writer):
        dataloader = iter(zip(cycle(self.labeled_loader), self.unlabeled_loader))
        self.model.train()

        running_loss = 0.
        last_loss = 0.

        for i, ((x_l, target_l), x_ul) in enumerate(dataloader):
            self.optimizer.zero_grad()
            outputs = self.model(x_l, x_ul)
            output_l = outputs["output_l"]
            output_ul = outputs["output_ul"]
            aux_outputs_ul = outputs["aux_outputs_ul"]
            target_ul = F.softmax(output_ul.detach(), dim=1)

            loss_l = supervised_loss(output_l, target_l, mode=self.sup_loss_mode)
            loss_ul = sum([ unsupervised_loss(output, target_ul, mode = self.unsup_loss_mode) for output in aux_outputs_ul]) / len(aux_outputs_ul)
            loss = loss_l + loss_ul * self.weight_ul

            loss.backward()
            self.optimizer.step()

            # report data
            running_loss += loss.item()
            if i % 100 == 0:
                last_loss = running_loss / 100
                print('  batch {} loss: {}'.format(i, last_loss))
                tb_x = epoch_idx * len(self.unlabeled_loader) + i
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    
    def eval_1epoch(self, ):
        return 0
    

  

#%% Tests

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

# # mode super
model_test = Model(mode='semi')

trainer_test = Trainer(model_test, labeled_dataloader, unlabeled_dataloader, None, mode="semi")
# trainer_test.train_super_1epoch(0, writer)
trainer_test.train_semi_1epoch(0, writer)







        

