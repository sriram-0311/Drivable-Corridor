import sys
sys.path.append(r'/home/ramesh.anu/AI Corridor/')

import torch
import torch.nn as nn
import cv2 as cv
from torchtest import assert_vars_change
from dataloaders.dataloaders import BDD
from models.cnn import CNN
from torch.utils.data import DataLoader

# assert if the variables change after the training
def AssertVariablesChange():
    root_dir = "/scratch/ramesh.anu/BDD/bdd100k/"
    bdd = BDD(root_dir)
    dataloader = DataLoader(bdd, batch_size=1, shuffle=True)
    batch = next(iter(dataloader))
    print(batch[0].shape)
    cnn = CNN()
    cnn.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print('Our list of parameters', [ np[0] for np in cnn.named_parameters() ])
    assert_vars_change(
        model=cnn,
        batch=batch,
        loss_fn=cnn.calculate_rmse_loss,
        optim=torch.optim.SGD(cnn.parameters(), lr=0.001),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

if __name__ == "__main__":
    AssertVariablesChange()