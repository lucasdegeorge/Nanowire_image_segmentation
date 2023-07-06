#%% 
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import date
import json

with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

print(device)

from dataloader import *
from model.model import * 
from trainer import * 
# from inference_scores import * 

def main():

    torch.cuda.empty_cache()
    mode = "semi"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(timestamp)

    with open("logs/logs_" + mode + "_" + str(timestamp) + ".txt","a") as logs :
        logs.write("START TRAINING IN MODE " + mode)
        logs.close()
    print("start training in mode " + mode)
    
    model = Model(mode=mode)
    model.to(device)

    with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
        arguments = json.load(f)
        batch_size = arguments["batch_size"]
        in_channels = arguments["model"]["in_channels"]
        print(batch_size)

    start_time = time.time()
    train_labeled_dataloader, eval_labeled_dataloader,  unlabeled_dataloader = get_dataloaders(in_channels, batch_size)
    print("dataloading took ", int(1000*(time.time()-start_time)), "ms")
    print(len(unlabeled_dataloader))
    trainer = Trainer(model, train_labeled_dataloader, unlabeled_dataloader, eval_labeled_dataloader, timestamp=timestamp)

    model_name = "C:/Users/lucas.degeorge/Documents/trained_models/" + "model_{}_{}.pth".format(mode, timestamp)

    trainer.train()
    print("end of training in mode " + mode)
    with open("logs/logs_" + mode + "_" + str(timestamp) + ".txt","a") as logs :
        logs.write("\n END OF TRAINING IN MODE " + mode + "- it took " + str(int(1000*(time.time()-start_time))) + "ms")
        logs.close()

    # meanIoU = compute_accuracy(model_name, eval_labeled_dataloader, True )
    return eval_labeled_dataloader

eval_dataloader = main()