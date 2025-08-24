import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from drivable_corridor.data.dataloaders import BDD
from drivable_corridor.models.cnn import CNN
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb
import numpy as np

torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120)

def main():
    # init a wandb project to store loss and learning rates
       # model training parameters
    learning_rate = 2e-4
    momentum = 0.9
    step_size = 10
    gamma = 0.5
    num_epochs = 100
    wandb.init(
        project="bdd100k",
        config={
            "learning_rate": learning_rate,
            "momentum": momentum,
            "step_size": step_size,
            "gamma": gamma,
            "num_epochs": num_epochs,
            "dataset" : "bdd100",
            "architecture" : "cnn"
        }
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    subset_size = 5000
    val_subset_size = 1000
    dataset_dir = "/scratch/ramesh.anu/BDD/bdd100k/"
    dataset = BDD(dataset_dir)
    subset_indices = np.random.randint(0, len(dataset), subset_size)
    dataset_subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(dataset_subset, batch_size=32, shuffle=True)
    val_dataset = BDD(dataset_dir, training=False)
    val_subset_indices = np.random.randint(0, len(val_dataset), val_subset_size)
    val_dataset_subset = Subset(val_dataset, val_subset_indices)
    val_dataloader = DataLoader(val_dataset_subset, batch_size=32, shuffle=True) 
    # print the shape of image and label from first sample in each batch
    for batch in dataloader:
        images, labels = batch
        print("Batch size --- ", images.size(0))
        print("Image shape --- ", images.size())
        print("Label shape --- ", labels.size())
        break

    model = CNN()
    model.to(device)

    # optimizers
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_adam = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer_adam, step_size=step_size, gamma=gamma)

    # if checkpoint is found load the weights and continue training
    try:
        model.load_state_dict(torch.load('../models/chkpoint.pth'))
        print("Checkpoint found, loading weights")
    except:
        pass

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(dataloader):
            images, labels = batch
            prediction = model(images)
            optimizer_adam.zero_grad()
            loss = model.bceloss(prediction, labels)
            loss.requires_grad_()
            loss.backward()
            optimizer_adam.step()
        # calculate the accuracy of the model on validation data
        accuracy = 0
        model.eval()
        for batch in tqdm(val_dataloader):
            images, labels = batch
            val_prediction = model(images)
            val_loss = model.bceloss(val_prediction, labels)
            accuracy += model.accuracy(val_prediction, labels)
        accuracy /= len(val_dataloader)
        print(f"in epoch {epoch} validation values predicted : ", torch.unique(val_prediction))
        print(f"Epoch : [{epoch+1}/{num_epochs}]; loss: {loss.item()}; lr: {scheduler.get_last_lr()[0]}; val_loss: {val_loss.item()}; val_accuracy: {accuracy}")
        wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "val_loss": val_loss.item(), "val_accuracy": accuracy})
        scheduler.step()

    # save the best weights
    torch.save(model.state_dict(), '../models/chkpoint_sgd_100_epocs.pth')

    # plot the loss and learning rate
    wandb.finish()

if __name__ == "__main__":
    main()