#%% 
import sys
import json
import torch
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/model") 

from preprocessing.dataloader import *
from losses import * 
from model import * 

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    trainer_arguments = arguments["trainer"]

#%% 

class Trainer:
    def __init__(self, model, labeled_loader, unlabeled_loader, eval_loader, arguments=trainer_arguments, device=device):
        self.model = model
        self.model.to(device)
        self.mode = self.model.mode

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
        assert self.mode == "super"

        dataloader = iter(self.labeled_loader)
        if not(self.model.training): self.model.train()

        running_loss = 0.
        last_loss = 0.

        for i, (x_l, target_l) in enumerate(dataloader):
            start_time = time.time()
            self.optimizer.zero_grad()
            output_l = self.model(x_l, None)["output_l"].to(device)

            loss = supervised_loss(output_l, target_l, mode=self.sup_loss_mode)
            loss.backward()
            self.optimizer.step()

            # report data
            running_loss += loss.item()
            if i % 10 == 0:
                if i==0: last_loss = running_loss
                else: last_loss = running_loss / 10
                # logs file 
                with open("logs.txt","a") as logs :
                    logs.write("\nEpoch : " + str(epoch_idx) + " - batch nb : "+str(i)+" -  in "+ str(int(1000*(time.time()-start_time))) + "ms, loss "+ str(last_loss))
                    logs.close()
                # tensorboard
                print('  batch {} loss: {}'.format(i, last_loss))
                tb_x = epoch_idx * len(self.unlabeled_loader) + i
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    
    def train_semi_1epoch(self, epoch_idx, tb_writer):
        assert self.mode == "semi"

        dataloader = iter(zip(cycle(self.labeled_loader), self.unlabeled_loader))
        if not(self.model.training): self.model.train()

        running_loss = 0.
        last_loss = 0.

        for i, ((x_l, target_l), x_ul) in enumerate(dataloader):
            start_time = time.time()
            self.optimizer.zero_grad()
            outputs = self.model(x_l, x_ul)
            output_l = outputs["output_l"].to(device)
            output_ul = outputs["output_ul"].to(device)
            aux_outputs_ul = outputs["aux_outputs_ul"]
            target_ul = F.softmax(output_ul.detach(), dim=1)

            loss_l = supervised_loss(output_l, target_l, mode=self.sup_loss_mode)
            loss_ul = sum([ unsupervised_loss(output, target_ul, mode = self.unsup_loss_mode) for output in aux_outputs_ul]) / len(aux_outputs_ul)
            loss = loss_l + loss_ul * self.weight_ul

            loss.backward()
            self.optimizer.step()

            # report data
            running_loss += loss.item()
            if i % 500 == 0:
                if i==0: last_loss = running_loss
                else: last_loss = running_loss / 500
                # logs file 
                with open("logs.txt","a") as logs :
                    logs.write("\nEpoch : " + str(epoch_idx) + " - batch nb : "+str(i)+" -  in "+ str(int(1000*(time.time()-start_time))) + "ms, loss "+ str(last_loss))
                    logs.close()
                # tensorboard
                print('  batch {} loss: {}'.format(i, last_loss))
                tb_x = epoch_idx * len(self.unlabeled_loader) + i
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    
    def eval_1epoch(self, epoch_idx):
        start_time = time.time()
        if self.model.training: self.model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for i, (val_x, val_target) in enumerate(self.eval_loader):
                val_output = self.model(val_x, None, eval=True)["output_l"].to(device)
                val_loss = F.cross_entropy(val_output, val_target)
                running_val_loss += val_loss
        val_loss = running_val_loss / (i + 1) 

        # report data 
        with open("logs.txt","a") as logs :
            logs.write("\nEpoch : " + str(epoch_idx) + " - Eval - in "+ str(int(1000*(time.time()-start_time))) + "ms, val_loss "+ str(val_loss.item()))
            logs.close()

        return  val_loss
    
    def train(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/trainer_{}_{}'.format(self.mode, timestamp))

        with open("logs.txt","a") as logs :
            logs.write("\n \n")
            logs.write("\nTraining - " + str(timestamp) + " - mode " + self.mode  + " - loss mode "  + self.sup_loss_mode + " " + self.unsup_loss_mode + "\n")
            logs.close()


        for epoch_idx in range(self.nb_epochs):
            print('EPOCH {}:'.format(epoch_idx))

            # train on one epoch 
            self.model.train(True)
            if self.mode == "semi":
                avg_train_loss = self.train_semi_1epoch(epoch_idx, writer)
            elif self.mode =="super":
                avg_train_loss = self.train_super_1epoch(epoch_idx, writer)

            # eval after the epoch
            self.model.eval()
            avg_val_loss = self.eval_1epoch(epoch_idx)

            # report data
            print('LOSS train {} eval {}'.format(avg_train_loss, avg_val_loss))
            writer.add_scalars('Training vs. Validation Loss', { 'Training' : avg_train_loss, 'Eval' : avg_val_loss }, epoch_idx)
            writer.flush()

        # save (best) models
        model_path = 'model_{}_{}'.format(self.mode, timestamp)
        torch.save(self.model.state_dict(), model_path)




#%% Tests

# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/test_trainer_{}'.format(timestamp))

# # # mode super
# model_test = Model(mode='super')

# trainer_test = Trainer(model_test, labeled_dataloader, unlabeled_dataloader, labeled_dataloader)
# # trainer_test.train_super_1epoch(0, writer)
# # trainer_test.train_semi_1epoch(0, writer)
# # trainer_test.eval_1epoch(0)
# trainer_test.train()
