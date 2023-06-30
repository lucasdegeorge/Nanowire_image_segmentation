#%% 

# part of the code is from: https://github.com/akanametov/pix2pix/blob/main/train.py

import torch
from torch import nn
from torch.utils.data import DataLoader
import time
from datetime import datetime, date
import json
from torch.utils.tensorboard import SummaryWriter
from pix2pix_dataloader import *

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

# dir 
image_dir = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"
mask_dir = "C:/Users/lucas.degeorge/Documents/Images/binary_masks"


class pix2pix_trainer:
    def __init__(self, generator, discriminator, criterions, lr, batch_size, timestamp=None) -> None:
        self.generator = generator
        self.discriminator = discriminator

        if timestamp is not None: self.timestamp = timestamp
        else: self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.g_criterions = criterions[0]
        self.d_criterions = criterions[1]

        self.g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.nb_epochs = 50

        self.train_dataloader = create_dataloader(image_dir, mask_dir, batch_size, shuffle=True, pin_memory=True)
        print(len(self.train_dataloader))

    def train_1epoch(self, epoch_idx, tb_writer):
        dataloader = iter(self.train_dataloader)

        if not(self.generator.training): self.generator.train()
        if not(self.discriminator.training): self.discriminator.train()

        last_loss = [0., 0.]
        running_loss = [0., 0.]

        for i, (real, x) in enumerate(dataloader):
            start_time = time.time()

            x = x.to(device)
            real = real.to(device)

            # generator's loss
            fake = self.generator(x)
            fake_pred = self.discriminator(fake, x)
            g_loss = self.g_criterions(fake, real, fake_pred)

            # discriminator's loss
            fake = self.generator(x).detach()
            fake_pred = self.discriminator(fake, x)
            real_pred = self.discriminator(real, x)
            d_loss = self.d_criterions(fake_pred, real_pred)

            # Generator`s backpropagation
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Discriminator`s backpropagation
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # report data
            running_loss[0] += g_loss.item()
            running_loss[1] += d_loss.item()
            if i % 2 == 0:
                if i==0: last_loss = running_loss
                else: last_loss = running_loss / 2
                # logs file 
                with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/data_augmentation/pix2pix/logs/logs_pix2pix_" + str(self.timestamp) + ".txt","a") as logs:
                    logs.write("\nEpoch : " + str(epoch_idx) + " - batch nb : "+str(i)+" -  in "+ str(int(1000*(time.time()-start_time))) + "ms, loss "+ str(last_loss))
                    logs.close()
                # tensorboard
                print('  batch {} loss: {}'.format(i, last_loss))
                tb_x = epoch_idx * len(self.train_dataloader) + i
                tb_writer.add_scalar('g_Loss/train', last_loss[0], tb_x)
                tb_writer.add_scalar('d_Loss/train', last_loss[1], tb_x)
                running_loss = 0.

        return last_loss

    def train(self):
        writer = SummaryWriter('runs/trainer_{}'.format(self.timestamp))
        with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/data_augmentation/pix2pix/logs/logs_pix2pix_" + str(self.timestamp) + ".txt","a") as logs :
            logs.write("\n \n")
            logs.write("\nTraining - " + str(self.timestamp) + "\n")
            logs.close()

        for epoch_idx in range(self.nb_epochs):
            print('EPOCH {}:'.format(epoch_idx))

            avg_train_loss = self.train_1epoch(epoch_idx, writer)

            # report data 
            print('LOSS generator {} discriminator {}'.format(avg_train_loss[0], avg_train_loss[1]))
            writer.add_scalars('Generator vs. Discriminator Loss', { 'Generator' : avg_train_loss[0], 'Discriminator' : avg_train_loss[1] }, epoch_idx)
            writer.flush()

            # save model 
            generator_path = 'C:/Users/lucas.degeorge/Documents/trained_models/pix2pix/pix2pix_generator_{}.pth'.format(self.timestamp)
            torch.save(self.generator.state_dict(), generator_path)
            discriminator_path = 'C:/Users/lucas.degeorge/Documents/trained_models/pix2pix/pix2pix_discriminator_{}.pth'.format(self.timestamp)
            torch.save(self.discriminator.state_dict(), discriminator_path)




