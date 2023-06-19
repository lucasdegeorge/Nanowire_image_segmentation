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

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/model") 

from dataloader import *
from model import * 
from trainer import * 
from inference_scores import * 

def main():

    torch.cuda.empty_cache()
    mode = "semi"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    with open("logs/logs_" + mode + "_" + str(timestamp) + ".txt","a") as logs :
        logs.write("START TRAINING IN MODE " + mode + "- 14/06/2023 ")
        logs.close()
    print("start training in mode " + mode)
    start_time = time.time()

    model = Model(mode=mode)
    model.to(device)

    with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
        arguments = json.load(f)
        batch_size = arguments["batch_size"]
        print(batch_size)

    train_labeled_dataloader, eval_labeled_dataloader,  unlabeled_dataloader = get_dataloaders(batch_size)
    print(len(unlabeled_dataloader))
    trainer = Trainer(model, train_labeled_dataloader, unlabeled_dataloader, eval_labeled_dataloader, timestamp=timestamp)

    model_name = "C:/Users/lucas.degeorge/Documents/trained_models/" + "model_{}_{}.pth".format(mode, timestamp)

    trainer.train()
    print("end of training in mode " + mode)
    with open("logs/logs_" + mode + "_" + str(timestamp) + ".txt","a") as logs :
        logs.write("\n END OF TRAINING IN MODE " + mode + "- 13/06/2023 - it took " + str(int(1000*(time.time()-start_time))) + "ms")
        logs.close()

    meanIoU = compute_accuracy(model_name, eval_labeled_dataloader, True )

main()