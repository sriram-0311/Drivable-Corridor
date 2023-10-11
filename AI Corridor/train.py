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

torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_dir = "/scratch/ramesh.anu/BDD/bdd100k/"
    dataset = BDD(dataset_dir)
    val_dataset = BDD(dataser_dir, training=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    # print the shape of image and label from first sample in each batch
    for batch in dataloader:
        images, labels = batch
        print("Batch size --- ", images.size(0))
        print("Image shape --- ", images.size())
        print("Label shape --- ", labels.size())
        break

    model = CNN()
    model.to(device)
    model.train()
    
    # model training parameters
    learning_rate = 1e-8
    momentum = 0.9
    step_size = 15
    gamma = 0.1
    num_epochs = 100

    # optimizers
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_adam = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer_adam, step_size=step_size, gamma=gamma)

    # if checkpoint is found load the weights and continue training
    try:
        model.load_state_dict(torch.load('checkpoints/chkpoint.pth'))
        print("Checkpoint found, loading weights")
    except:
        pass

    for epoch in range(num_epochs):
        for batch in dataloader:
            images, labels = batch
            prediction = model(images)
            optimizer.zero_grad()
            loss = model.calculate_rmse_loss(prediction, labels)
            loss.requires_grad_()
            loss.backward()
            optimizer.step()
        scheduler.step() # learning rate decay
        # calculate the accuracy of the model on validation data
        correct = 0
        total = 0
        for batch in val_dataloader:
            images, labels = batch
            prediction = model(images)
            val_loss = model.calculate_rmse_loss(prediction, labels)
            total += labels.shape(0)
            correct += (prediction == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch : [{epoch+1}/{num_epochs}]; loss: {loss.item()}; lr: {scheduler.get_last_lr()[0]}; val_loss: {val_loss.item()}; accuracy: {accuracy}")
        wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "validation loss": val_loss.item(), "accuracy": accuracy})

    # save the best weights
    torch.save(model.state_dict(), 'checkpoints/chkpoint_sgb.pth')

    # plot the loss and learning rate
    wandb.finish()

if __name__ == "__main__":
    main()