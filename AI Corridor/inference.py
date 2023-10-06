import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from models.cnn import CNN
from torch.utils.data import DataLoader
from dataloaders.dataloaders import BDD
from torchvision import transforms

def main():
    #define cuda device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = CNN()
    model.load_state_dict(torch.load('checkpoints/chkpoint.pth'))
    model.eval()
    model.to(device)

    # load the image
    image = cv.imread('/scratch/ramesh.anu/BDD/bdd100k/images/10k/test/ac517380-00000000.jpg', 0)
    image = cv.resize(image, (652, 360)).T
    image = transforms.ToTensor()(image).to(device)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    
    # predict the output
    output = model(image)
    output = output.cpu().detach().numpy()
    output = np.squeeze(output)
    output = np.where(output > 0, 255, 0)
    
    # plot the output
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().squeeze().T, cmap='gray')
    ax[1].imshow(output.T, cmap='gray')
    plt.savefig("prediction.png")

if __name__ == "__main__":
    main()