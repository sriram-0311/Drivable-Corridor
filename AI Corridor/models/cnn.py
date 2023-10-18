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
        self.conv10 = nn.Conv2d(64, 64, padding=(2,0), kernel_size=(5,2))
        self.conv11 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv13 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv14 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv15 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv16 = nn.Conv2d(32, 16, 5, padding=2)
        self.conv17 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv18 = nn.Conv2d(8, 1, 3, padding=1)

        # define the pooling layers
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

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
        self.dropout = nn.Dropout(0.2)

        # sigmoid activation layer
        self.sigmoid = nn.Sigmoid()

    # define the forward pass
    def forward(self, x):
        # encoder
        # block 1
        x = self.bn1(x)
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1))
        p1 = self.pool(c2)

        # block 2
        c3 = self.dropout(self.relu(self.conv3(p1)))
        c4 = self.dropout(self.relu(self.conv4(c3)))
        c5 = self.dropout(self.relu(self.conv5(c4)))
        p2 = self.pool(c5)

        # block 3
        c6 = self.dropout(self.relu(self.conv6(p2)))
        c7 = self.dropout(self.relu(self.conv7(c6)))
        p3 = self.pool(c7)

        # block 4
        c8 = self.dropout(self.relu(self.conv8(p3)))
        c9 = self.dropout(self.relu(self.conv9(c8)))
        p4 = self.pool(c9)

        # decoder
        # block 1
        up1 = self.deconv1(p4)
        c10 = self.dropout(self.relu(self.conv10(up1)))
        c11 = self.dropout(self.relu(self.conv11(c10)))

        # block 2
        up2 = self.deconv2(c11+c9)
        c12 = self.dropout(self.relu(self.conv12(up2)))
        c13 = self.dropout(self.relu(self.conv13(c12+c7)))

        # block 3
        up3 = self.deconv3(c13)
        c14 = self.dropout(self.relu(self.conv14(up3)))
        c15 = self.dropout(self.relu(self.conv15(c14+c5)))
        c16 = self.dropout(self.relu(self.conv16(c15)))

        # block 4
        up4 = self.deconv4(c16)
        c17 = self.relu(self.conv17(up4+c2))
        c18 = self.conv18(c17)

        return c18
    
    # define the loss function
    def calculate_rmse_loss(self, prediction, target):
        assert prediction.shape[0] == target.shape[0]
        assert prediction.shape[1] == target.shape[1]
        assert prediction.shape[2] == target.shape[2]
        assert prediction.shape[3] == target.shape[3]
        
        mse_loss = nn.MSELoss()
        loss = mse_loss(prediction, target)
        return loss

    # define a loss function to calculate the Dice loss
    def CalculateDiceLoss(self, prediction, target):
        smooth = 1e-7
        # print(torch.unique(prediction))
        # apply sigmoid on each pixel to convert pixel values to be between 0 and 1
        # predicted_probabilities = torch.sigmoid(prediction)
        # print(torch.unique(predicted_probabilities))
        prediction_flat = prediction.view(-1)
        target_flat = target.view(-1)
        intersection = (prediction_flat * target_flat).sum()
        # print(torch.mean(1. - ((2. * intersection + smooth) / (prediction_flat.sum() + target_flat.sum() + smooth))))
        return  (1. - ((2. * intersection + smooth) / (prediction_flat.sum() + target_flat.sum() + smooth)))/prediction.shape[0]

    # define bce loss with logits
    def bceloss(self, prediction, target):
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(prediction, target)
    
    # define calculate accuracy of model using IoU metric
    def accuracy(self, prediction, target):
        prediction = torch.where(torch.sigmoid(prediction) > 0.5, 1, 0)
        intersection = (prediction * target).sum()
        union = (prediction + target).sum()
        return (intersection + 1e-7) / ((union + 1e-7))
        
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

