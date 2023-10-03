import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataloaders.dataloaders import BDD
from models.cnn import CNN
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import tqdm
import wandb

def main():
    # init a wandb project to store loss and learning rates
    wandb.init(
        project="bdd100k",
        config={
            "learning_rate": 1e-6,
            "momentum": 0.9,
            "step_size": 30,
            "gamma": 0.1,
            "num_epochs": 200,
            "dataset" : "bdd100",
            "architecture" : "cnn"
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = "/scratch/ramesh.anu/BDD/bdd100k/"
    dataset = BDD(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # print the shape of image and label from first sample in each batch
    for batch in dataloader:
        images, labels = batch
        print("Batch size --- ", images.size(0))
        print("Image shape --- ", images.size())
        print("Label shape --- ", labels.size())
        break

    model = CNN()
    model.to(device)
    
    # model training parameters
    learning_rate = 1e-6
    momentum = 0.9
    step_size = 30
    gamma = 0.1
    num_epochs = 200

    # optimizers
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(num_epochs):
        for batch in dataloader:
            images, labels = batch
            # print(images.get_device())
            # print(labels.get_device())
            prediction = model(images)
            loss = model.calculate_rmse_loss(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
            optimizer.step()
        scheduler.step() # learning rate decay
        print(f"Epoch : [{epoch+1}/{num_epochs}]; loss: {loss.item()}; lr: {scheduler.get_last_lr()[0]}")
        wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

    # save the best weights
    torch.save(model.state_dict(), 'checkpoints/chkpoint.pth')

    # plot the loss and learning rate
    wandb.finish()

if __name__ == "__main__":
    main()