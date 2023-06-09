#%% 
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from datetime import datetime

from DCGAN_model import *
from DCGAN_dataloader import *

device = torch.device("cuda")

images_path = "C:/Users/lucas.degeorge/Documents/Images/labeled_images"

nb_epochs = 20 
lr = 0.0002 
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers
image_size = 1024 
batch_size = 4

# dataloaders 
data_path = "C:/Users/lucas.degeorge/Documents/Images/resized_images/combined_data"
dataloader = get_dataloader(data_path, batch_size=batch_size, shuffle=True, pin_memory=True)

# Generator
netG = Generator().to(device)
netG.apply(weights_init) # to get weights with mean=0 and stdev=0.02

# Discriminator
netD = Discriminator().to(device)
netD.apply(weights_init)

# loss functions
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, noise_channels, 1, 1, device=device)
real_label = 1.
fake_label = 0.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


def train():
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("Starting Training Loop...")
    for epoch in range(nb_epochs):

        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()

            ## Train the discriminator with the real data
            real_cpu = data.to(device)
            ba_size = real_cpu.size(0)

            label = torch.full((ba_size,), real_label, dtype=torch.float, device=device) # ground truth for real data
            output = netD(real_cpu)
            print(output.shape)
            output = output.view(-1)
            print(output.shape)  # prediction by the discriminator on real image
            errD_real = criterion(output, label)

            errD_real.backward()
            D_x = output.mean().item()

            ## Generate fake data and train the discriminator with 
            noise = torch.randn(ba_size, noise_channels, 1, 1, device=device) # generates latent vectors (noise) for the generator
            generated_image = netG(noise)  # image generated by the generator
            label.fill_(fake_label) # ground truth for fake data

            output = netD(generated_image.detach()).view(-1)  # prediction by the discriminator on fake/generated image

            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            ## Train the generator (maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  
            output = netD(generated_image).view(-1) # G must deceive D. The data generated must be considered real by D

            errG = criterion(output, label) 
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, nb_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == nb_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        
            iters += 1

        # save models after each epoch 
        G_path = "C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/data_augmentation/GAN_4ch/saved_models/generator_{}_epoch{}".format(timestamp, epoch)
        D_path = "C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/data_augmentation/GAN_4ch/saved_models/discriminator_{}_epoch{}".format(timestamp, epoch) 
        torch.save(netG.state_dict(), G_path)
        torch.save(netD.state_dict(), D_path)

    return img_list, G_losses, D_losses

train()