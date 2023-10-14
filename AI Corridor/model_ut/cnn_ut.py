import sys
sys.path.append(r'/home/ramesh.anu/Drivable-Corridor/AI Corridor/')

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
    dataloader = DataLoader(bdd, batch_size=32, shuffle=True)
    batch = next(iter(dataloader))
    cnn = CNN()
    cnn.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    print("number of model params ",sum(p.numel() for p in cnn.parameters() if p.requires_grad))
    # print range of values and number of values in between those in batch[0]
    # print("range :", torch.max(batch[0]), torch.min(batch[0]))
    # print("number of unique values : ", torch.unique(batch[0]))
    assert_vars_change(
        model=cnn,
        batch=batch,
        loss_fn=cnn.bceloss,
        optim=torch.optim.SGD(cnn.parameters(), lr=0.05),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

# print the gradients of the model parameters for one train step and check if it is changing
def AssertNonZeroGradients():
    root_dir = "/scratch/ramesh.anu/BDD/bdd100k/"
    bdd = BDD(root_dir)
    dataloader = DataLoader(bdd, batch_size=1, shuffle=True)
    batch = next(iter(dataloader))
    # print(batch[0].shape)
    cnn = CNN()
    cnn.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # print('Our list of parameters', [ np[0] for np in cnn.named_parameters() ])
    optimizer_adam = torch.optim.Adam(cnn.parameters(), lr=0.001)
    optimizer_adam.zero_grad()
    cnn.train()
    pred = cnn(batch[0])
    print("range :", torch.max(pred), torch.min(pred))
    print("number of unique values : ", torch.unique(pred))
    loss = cnn.CalculateDiceLoss(pred, batch[1])
    print(loss)
    loss.backward()
    optimizer_adam.step()
    # print('Gradients - ', [np.grad for np in cnn.parameters()])
    assert any([np.grad is not None for np in cnn.parameters()])

if __name__ == "__main__":
    AssertVariablesChange()
    # AssertNonZeroGradients()