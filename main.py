#%% 
import sys
import time
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/model") 

from preprocessing.dataloader import *
from model import * 
from trainer import * 

def main():

    torch.cuda.empty_cache()

    for mode in ["semi", "super"]:

        with open("logs.txt","a") as logs :
            logs.write("START TRAINING IN MODE " + mode + "- 13/06/2023 ")
            logs.close()
        print("start training in mode " + mode)
        start_time = time.time()

        model = Model(mode=mode)
        model.to(device)

        with open("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/parameters.json", 'r') as f:
            arguments = json.load(f)
            batch_size = arguments["batch_size"]

        train_labeled_dataloader, eval_labeled_dataloader,  unlabeled_dataloader = get_dataloaders(batch_size)
        trainer = Trainer(model, train_labeled_dataloader, unlabeled_dataloader, eval_labeled_dataloader)

        trainer.train()
        print("end of training in mode " + mode)
        with open("logs.txt","a") as logs :
            logs.write("END OF TRAINING IN MODE " + mode + "- 13/06/2023 - it took " + str(int(1000*(time.time()-start_time))) + "ms")
            logs.close()
        print()


main()