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
    test_images = "images/10k/test/"

    test_files = os.listdir(rootd + test_images)

    # load the model
    model = CNN()
    model.load_state_dict(torch.load('checkpoints/chkpoint_adam.pth'))
    model.eval()
    model.to(device)

    # load the image
    test_image = np.random.choice(test_files)
    image = cv.imread(rootd+test_images+test_image, 0)
    image = cv.resize(image, (652, 360)).T
    image = transforms.ToTensor()(image).to(device)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    
    # predict the output
    output = model(image)
    print(torch.unique(output))
    output = output.cpu().detach().numpy()
    output = np.squeeze(output)
    # output = np.where(output > 0, 255, 0)
    
    # plot the output
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().squeeze().T, cmap='gray')
    ax[1].imshow(output.T, cmap='gray')
    plt.savefig("prediction.png")

if __name__ == "__main__":
    main()