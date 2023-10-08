# create a cnn model

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120)

class CNN(nn.Module):
    def __init__(self):
        # create a module of conv layer and relu layer
        super(CNN, self).__init__()

        # define the convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv6 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv11 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv13 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv14 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv15 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv16 = nn.Conv2d(32, 16, 5, padding=2)
        self.conv17 = nn.Conv2d(16, 16, kernel_size=(5,9))
        self.conv18 = nn.Conv2d(16, 1, 3, padding=1)

        # define the pooling layers
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # define the relu layers
        self.relu = nn.ReLU()

        # define the deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(16, 16, 2, stride=2)

        # define the batch normalization layers
        self.bn1 = nn.BatchNorm2d(1)

        # define the dropout layers
        self.dropout = nn.Dropout(0.5)

        # softmax activation layer
        self.softmax = nn.Softmax(dim=1)

    # define the forward pass
    def forward(self, x):
        # encoding block
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool1(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool1(x)

        x = self.conv8(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv9(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool1(x)

        # decoding block
        x = self.deconv1(x)
        x = self.conv10(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv11(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.deconv2(x)
        x = self.conv12(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv13(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.deconv3(x)
        x = self.conv14(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv15(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv16(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.deconv4(x)
        x = self.conv17(x)
        x = self.relu(x)
        x = self.conv18(x)
        x = self.relu(x)
        
        # activation function
        # x = self.softmax(x)

        return x
    
    # define the loss function
    def calculate_rmse_loss(self, prediction, target):
        assert prediction.shape[0] == target.shape[0]
        assert prediction.shape[1] == target.shape[1]
        assert prediction.shape[2] == target.shape[2]
        assert prediction.shape[3] == target.shape[3]
        
        mse_loss = nn.MSELoss()
        loss = torch.sqrt(mse_loss(prediction, target))
        return loss

    # define a loss function to calculate the Dice loss
    def CalculateDiceLoss(self, prediction, target):
        assert prediction.shape[0] == target.shape[0]
        assert prediction.shape[1] == target.shape[1]
        assert prediction.shape[2] == target.shape[2]
        assert prediction.shape[3] == target.shape[3]

        # calculate the dice loss
        dice_loss = 0
        smooth = 1e-7
        for i in range(prediction.shape[0]):
            intersection = torch.sum(prediction * target)
            union = torch.sum(prediction) + torch.sum(target)
            dice_score = (2 * intersection + smooth) / (union + smooth)
            dice_loss += (1 - dice_score)
        dice_loss /= prediction.shape[0]

        return dice_loss
        
# main function
def main():
    # create a random input of shape (100, 1, 652, 360) to test the model
    input = torch.rand(100, 1, 652, 360)

    model = CNN()
    # model.to(device)

    # get the output of the model
    output = model(input)

    # print the output shape
    print(output.shape)

if __name__ == '__main__':
    main()

