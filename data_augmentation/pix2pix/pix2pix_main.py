#%% 
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime
import json
import sys

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation")

from data_augmentation.pix2pix.generator import UnetGenerator
from data_augmentation.pix2pix.discriminator import ConditionalDiscriminator
from data_augmentation.pix2pix.pix2pix_loss import GeneratorLoss, DiscriminatorLoss
from data_augmentation.pix2pix.pix2pix_trainer import *
from dataloader import *

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

def main():

    torch.cuda.empty_cache()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("start training")
    start_time = time.time()

    batch_size = 2
    lr = 0.0002
    in_channels = 1

    generator = UnetGenerator().to(device)
    discriminator = ConditionalDiscriminator().to(device)
    criterions = [ GeneratorLoss(alpha=100), DiscriminatorLoss() ]

    trainer = pix2pix_trainer(generator, discriminator, criterions, lr, batch_size, in_channels, timestamp=timestamp)
    trainer.train()

    print("end of training")

main()

