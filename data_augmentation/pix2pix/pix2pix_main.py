#%% 
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime
import json
import sys

from generator import UnetGenerator
from discriminator import ConditionalDiscriminator
from pix2pix_loss import GeneratorLoss, DiscriminatorLoss
from pix2pix_trainer import *

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

def main():

    torch.cuda.empty_cache()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("start training")
    start_time = time.time()

    batch_size = 8
    lr = 0.0002

    generator = UnetGenerator().to(device)
    discriminator = ConditionalDiscriminator().to(device)
    criterions = [ GeneratorLoss(alpha=100), DiscriminatorLoss() ]

    trainer = pix2pix_trainer(generator, discriminator, criterions, lr, batch_size, timestamp=timestamp)
    trainer.train()

    print("end of training")

main()

