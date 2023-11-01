# create a cnn model

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120)

# feature fusion network
class FFN(nn.Module):
    def __init__(self, spatial_features, context_features):
        super(FFN, self).__init__()
        self.if1 = spatial_features.shape[1]
        self.if2 = context_features.shape[1]
        
        # define the conv + bn + relu unit
        self.conv1 = nn.Conv2d((self.if1+self.if2), 512, kernel_size=3)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()

        self.avgpool = nn.AvgPool2d(kernel_size=512)

        self.1x1conv = nn.Conv2d(512, 512, kernel_size=1)
        # self.1x1conv2 = nn.Conv2d(512, 512, kernel_size=1)
        

    def forward(self, x, y):
        # concatenate the spatial and context features
        x = torch.cat((x, y), dim=1)
        x = self.relu(self.bn(self.conv1(x)))
        y = self.avgpool(x)
        y = self.relu(self.1x1conv(x))
        y = self.1x1conv(x)
        y = torch.sigmoid(x)
        
        return (x*y + x)

# attention refinement network
class ARN(nn.Module):
    def __init__(self, in_features):
        super(ARN, self).__init__()
        # define the global average pooling layer using avgpool2d
        self.avgpool = nn.AvgPool2d(kernel_size=in_features.shape[2:])
        self.conv1 = nn.Conv2d(in_features.shape[1], in_features.shape[1], kernel_size=1)
        self.bn = nn.BatchNorm2d(in_features.shape[1])
    
    def forward(self, x):
        # get the global average pooling layer
        y = self.avgpool(x)
        # get the conv layer
        y = self.conv1(y)
        # get the batch normalization layer
        y = self.bn(y)
        # get the sigmoid activation layer
        y = torch.sigmoid(y, dim=-1)
        return (x*y)

class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
         model_res = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # remove the last layer from the resnet model and freeze the parameters
        for param in model_res.parameters():
            param.requires_grad = False
        self.resnet = model_res
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        
    def forward(self, x):
        # should call the ARN module here
        # out = self.resnet(x)
        y = self.conv1(x)       # 2x down
        y = self.bn1(x)
        y = self.relu(x)
        y = self.maxpool(x)     # 4x down
        y = self.layer1(x)
        y = self.layer2(x)      # 8x down
        y = self.layer3(x)      # 16x down
        
        out1 = self.ARN(y)
        
        y = self.layer4(y)      # 32x down

        out2 = self.ARN(y)

        # feature_map1 = self.layer3(x)
        # arm1 = self.ARN(feature_map1)
        
        # feature_map2 = nn.Sequential(*list(model_res.children())[-2]).in_features
        # arm2 = self.ARN(feature_map2)
        
        # how to combine the 2 feature maps?
        out = torch.cat((out1, out2), dim=1)

        return (out)

class SpatialPath(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(SpatialPath, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=2, padding=1)
        
        # define the relu layers
        self.relu = nn.ReLU()
        
        # define the batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        out = self.relu(self.bn3(self.conv3(x)))
        return out

class Bisenet(nn.Module):
    def __init__(self):
        # create a module of conv layer and relu layer
        super(Bisenet, self).__init__()

        
    # define the forward pass
    def forward(self, x):
        
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