#%% 
import sys
import json
import torch
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, date
import time
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

c2n = True

# Device configuration
with open("parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

# from dataloader import *
from dataloader_sep import *
from losses import * 
from model.model import * 

#%% 

class Trainer:
    def __init__(self, model, labeled_loader, unlabeled_loader, eval_loader, arguments=arguments, device=device, timestamp=None):
        self.model = model
        self.model.to(device)
        self.mode = self.model.mode

        if timestamp is not None: self.timestamp = timestamp
        else: self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # supervised loss
        self.sup_loss_mode = arguments["trainer"]["sup_loss"]
        
        # unsupervised loss
        self.unsup_loss_mode = arguments["trainer"]["unsup_loss"]
        self.weight_ul_max = arguments["trainer"]["weight_ul_max"]
         
        # optimizer 
        if arguments["trainer"]["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=arguments["trainer"]["optimizer_args"]["lr"], momentum=arguments["trainer"]["optimizer_args"]["momentum"])
        elif arguments["trainer"]["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=arguments["trainer"]["optimizer_args"]["lr"])
        else:
            raise ValueError("optimizer has an invalid value. Must be in ['sgd']")
        
        # scheduler
        if arguments["trainer"]["scheduler"] == "OneCycleLR":
            self.scheduler = OneCycleLR(self.optimizer, max_lr = 1e-2, steps_per_epoch = 20000, epochs = arguments["trainer"]["nb_epochs"], anneal_strategy = 'cos')
        elif arguments["trainer"]["scheduler"] == "CosineAnnealingLR":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max = 40, eta_min = 1e-5)
        elif arguments["trainer"]["scheduler"] == "CosineAnnealingWarmRestarts":
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, t_0=5, T_max = 40, eta_min = 1e-6)
        else:
            self.scheduler = None
            # raise ValueError("scheduler has an invalid value. Must be in ['OneCycleLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts]")
        
        self.nb_epochs = arguments["trainer"]["nb_epochs"]
        if self.mode == 'semi':
            self.iter_per_epoch = len(unlabeled_loader) # assuming that len(unlabeled) > len(labeled)
            self.rampup_length = self.nb_epochs * self.iter_per_epoch

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

            x_l = x_l.to(device)
            target_l = target_l.to(device)

            self.optimizer.zero_grad()
            output_l = self.model(x_l, None)["output_l"].to(device)

            loss = supervised_loss(output_l, target_l, mode=self.sup_loss_mode)
            loss.backward()
            self.optimizer.step()

            # report data
            running_loss += loss.item()
            if i % 100 == 0:
                if i==0: last_loss = running_loss
                else: last_loss = running_loss / 100
                # logs file 
                with open("logs/logs_" + self.mode + "_" + str(self.timestamp) + ".txt","a") as logs :
                    logs.write("\nEpoch : " + str(epoch_idx) + " - batch nb : "+str(i)+" -  in "+ str(int(1000*(time.time()-start_time))) + "ms, loss "+ str(last_loss))
                    logs.close()
                # tensorboard
                print('  batch {} loss: {}'.format(i, last_loss))
                tb_x = epoch_idx * len(self.labeled_loader) + i
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

            x_l = x_l.to(device)
            target_l = target_l.to(device)
            x_ul = x_ul.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(x_l, x_ul)
            output_l = outputs["output_l"].to(device)
            output_ul = outputs["output_ul"].to(device)
            aux_outputs_ul = outputs["aux_outputs_ul"]
            target_ul = F.softmax(output_ul.detach(), dim=1)

            loss_l = supervised_loss(output_l, target_l, mode=self.sup_loss_mode)
            loss_ul = sum([ unsupervised_loss(output, target_ul, mode = self.unsup_loss_mode) for output in aux_outputs_ul]) / len(aux_outputs_ul)
            w_u = 1 #  weight_ramp_up(self.iter_per_epoch * epoch_idx + i, self.rampup_length, self.weight_ul_max)
            loss = loss_l + loss_ul * w_u

            loss.backward()
            self.optimizer.step()

            # report data
            running_loss += loss.item()
            if i % 1000 == 0:
                if i==0: last_loss = running_loss
                else: last_loss = running_loss / 1000
                # logs file 
                with open("logs/logs_" + self.mode + "_" + str(self.timestamp) + ".txt","a") as logs :
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
                val_x = val_x.to(device)
                val_target = val_target.to(device)

                val_output = self.model(val_x, None, eval=True)["output_l"].to(device)
                val_loss = eval_loss(val_output, val_target, mode=self.sup_loss_mode)
                running_val_loss += val_loss
        val_loss = running_val_loss / (i + 1) 

        # report data 
        with open("logs/logs_" + self.mode + "_" + str(self.timestamp) + ".txt","a") as logs :
            logs.write("\nEpoch : " + str(epoch_idx) + " - Eval - in "+ str(int(1000*(time.time()-start_time))) + "ms, val_loss "+ str(val_loss.item()))
            logs.close()

        return  val_loss
    
    def train(self):
        best_val_loss = 1e20
        writer = SummaryWriter('runs/trainer_{}_{}'.format(self.mode, self.timestamp))
        with open("logs/logs_" + self.mode + "_" + str(self.timestamp) + ".txt","a") as logs :
            logs.write("\n \n")
            logs.write("\nTraining - " + str(self.timestamp) + " - mode " + self.mode  + " - loss mode "  + self.sup_loss_mode + " " + self.unsup_loss_mode + "\n")
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

            # scheduler 
            if self.scheduler is not None:
                self.scheduler.step()

            # report data
            print('LOSS train {} eval {}'.format(avg_train_loss, avg_val_loss))
            writer.add_scalars('Training vs. Validation Loss', { 'Training' : avg_train_loss, 'Eval' : avg_val_loss }, epoch_idx)
            writer.flush()

            # save (best) models
            if c2n:
                model_path = 'C:/Users/lucas.degeorge/Documents/trained_models/model_{}_{}_epoch{}.pth'.format(self.mode, self.timestamp, epoch_idx)
            else:
                model_path = 'trained_models/model_{}_{}_epoch{}.pth'.format(self.mode, self.timestamp, epoch_idx)
            torch.save(self.model.state_dict(), model_path)
            if avg_val_loss < best_val_loss:
                if c2n: model_path = 'C:/Users/lucas.degeorge/Documents/trained_models/model_{}_{}_best.pth'.format(self.mode, self.timestamp)
                else: model_path = 'trained_models/model_{}_{}_best.pth'.format(self.mode, self.timestamp)
                torch.save(self.model.state_dict(), model_path)
                print("new best epoch: ", epoch_idx)
                best_val_loss = avg_val_loss





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
