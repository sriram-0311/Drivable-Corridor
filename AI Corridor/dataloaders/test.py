import torch
from torchvision import transforms
import cv2 as cv
import numpy as np
import os

rootd =  "/scratch/ramesh.anu/BDD/bdd100k/"
training_path = "images/10k/train/"
labels_path = "labels/drivable/colormaps/train/"
masks_path = "labels/drivable/masks/train/"

# read a random image from masks and random image from colormaps
masks_files = os.listdir(rootd + masks_path)
colormap_files = os.listdir(rootd + labels_path)

print("Number of masks: ", len(masks_files))
print("Number of colormaps: ", len(colormap_files))

# get a random image from masks
random_mask = np.random.choice(masks_files)
random_colormap = np.random.choice(colormap_files)

# read the images
mask = cv.imread(rootd + masks_path + random_mask, 0)
colormap = cv.imread(rootd + labels_path + random_colormap, 0)

# print the values in the images and min and max and unique values in each image
print("Mask ---- ",mask)
print("Shape of Mask ---- ", mask.shape)
print("color map ----", colormap)
print("unique values in mask ----",np.unique(mask))
print("unique values in colormaps ---- ", np.unique(colormap))

# convert the mask to a tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
mask_tensor = transform(mask)

# print the values in the tensor
print("Mask tensor ---- ", mask_tensor)


two_indices = np.where(mask == 2)
one_indices = np.where(mask == 1)
mask[mask == 0] = 1
mask[one_indices] = 0
mask[two_indices] = 0

# print the mask and it's unique values
print("mask ---- ", mask)
print("unique values in mask ---- ", np.unique(mask))

# create a new rgb image and make pixels where mask has value 2 as green value 1 as blue and value 0 as red
rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
rgb[mask == 2] = [0, 255, 0]
rgb[mask == 1] = [0, 0, 255]
rgb[mask == 0] = [255, 0, 0]
rgb = cv.resize(rgb, (652,360))

# save the image to sample3
cv.imwrite("sample_mask_2.png", rgb)


