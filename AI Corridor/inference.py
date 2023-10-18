import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from models.cnn import CNN
from torch.utils.data import DataLoader
from dataloaders.dataloaders import BDD
from torchvision import transforms
import os

def main():
    #define cuda device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rootd =  "/scratch/ramesh.anu/BDD/bdd100k/"
    test_images = "images/10k/train/"

    test_files = os.listdir(rootd + test_images)

    # load the model
    model = CNN()
    model.load_state_dict(torch.load('checkpoints/chkpoint_sgb_50_epocs.pth'))
    model.eval()
    model.to(device)

    # load image using the bdd dataloader
    bdd_dataset = BDD(rootd)
    random_choice_image_index = np.random.choice(range(len(bdd_dataset)))
    img, lbl = bdd_dataset[random_choice_image_index]
    img = img.reshape(1,img.shape[0], img.shape[1], img.shape[2])
    # print(torch.unique(img))
    
    # predict the output
    output = model(img)
    # apply inverse sigmoid on the output
    print("test 1 ",torch.unique(output))
    output = torch.sigmoid(output)
    print("test 2",torch.unique(output))
    output = output.cpu().detach().numpy()
    output = np.squeeze(output)
    output = np.where(output > 0.5, 255, 0)
    print(np.unique(output))
    
    # plot the output
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(img.cpu().detach().numpy().squeeze().T, cmap='gray')
    ax[1].imshow(output.T)
    plt.savefig("prediction.png")

if __name__ == "__main__":
    main()