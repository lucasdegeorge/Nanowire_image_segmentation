from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append("C:/Users/lucas.degeorge/Documents/GitHub/Nanowire_image_segmentation/model") 

from preprocessing.dataloader import *
from model import * 
from trainer import * 

def main(arguments):

    mode = arguments["mode"]
    model = Model(mode=mode)
    model.to(device)

    train_labeled_dataloader, eval_labeled_dataloader,  unlabeled_dataloader = get_dataloaders()
    trainer_test = Trainer(model, train_labeled_dataloader, unlabeled_dataloader, eval_labeled_dataloader)

    trainer_test.train()

if __name__=='__main__':
    main()