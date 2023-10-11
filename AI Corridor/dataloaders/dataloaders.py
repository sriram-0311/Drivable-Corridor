from torch.utils import data
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BDD(data.Dataset):
    def __init__(self, root_dir, training=True):
        self.root_dir = root_dir
        self.training_path = "images/10k/train/"
        self.validation_path  = "images/10k/val"
        self.labels_path = "labels/drivable/masks/train/"
        self.train_data = training

        self.training_image_names = [name.split(".")[0] for name in os.listdir(self.root_dir + self.training_path)]
        self.labels = [name.split(".")[0] for name in os.listdir(self.root_dir + self.labels_path)]
        # print("Number of labels: ", len(self.labels))
        # print("Number of training images: ", len(self.training_image_names))

        # create a new list of elements that are in both lists
        self.training_image_with_labels = list(set(self.training_image_names).intersection(self.labels))
        # print("Number of training images with labels: ", len(self.training_image_with_labels))
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])]
        )

        # if training data is asked
        if self.train_data:
            self.files = []
            for name in self.training_image_with_labels:
                image_name = os.path.join(self.root_dir + self.training_path, name + ".jpg")
                label_name = os.path.join(self.root_dir + self.labels_path, name + ".png")
                self.files.append({
                    "image": image_name,
                    "label": label_name
                })

        # if validation data is asked
        else:
            self.files = []
            self.validation_image_names = [names.split(".")[0] for name in os.listdir(self.root_dir) + self.validation_path]
            self.validation_images_with_labels = list(set(self.validation_image_names).intersection(self.labels))
            for name in self.validation_images_with_labels:
                image_name = os.path.join(self.root_dir + self.validation_path, name + ".jpg")
                label_name = os.path.join(self.root_dir + self.labels_path, name + ".png")
                self.files.append({
                    "image": image_name,
                    "label": label_name
                })
        

    def __len__(self):
        return len(self.training_image_with_labels)
    
    def __getitem__(self, index):
        file_name = self.files[index]
        image = cv.imread(file_name["image"], 0)
        image = cv.resize(image, (652, 360)).T
        label = cv.resize(cv.imread(file_name["label"], 0), (652, 360)).T
        # make the pixels where label value = 2 as 0 and where value = 0 as 1
        two_indices = np.where(label == 2)
        one_indices = np.where(label == 1)
        label[one_indices] = 0
        label[label == 0] = 1
        label[two_indices] = 0
        image = self.transform(image)
        label = torch.from_numpy(np.reshape(label, (1,652,360))).float()
        return image.to(device).requires_grad_(), label.to(device).requires_grad_()
    
def test():
    root_dir = "/scratch/ramesh.anu/BDD/bdd100k/"
    bdd = BDD(root_dir)
    _, ax = plt.subplots(20, 3)
    # make the images in the plot bigger and margins thin
    plt.rcParams["figure.figsize"] = (20, 20)
    # reduce the distance between the images in the plot
    plt.subplots_adjust(wspace=0.1, hspace=0.1)


    # print(len(bdd))
    # get 20 random non-consecutive samples from dataset
    samples = np.random.choice(len(bdd), 20, replace=False)
    for i, sample in enumerate(samples):
        image, label = bdd[sample]
        print(image.shape)
        print(label.shape)
        # check if image is scaled properly
        # print(image.max())
        ax[i, 0].imshow(image.cpu().detach().squeeze(0).T, cmap="gray")
        ax[i, 1].imshow(label.cpu().detach().squeeze(0).T, cmap="gray")
        # overlay label on top of image
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
        ax[i, 2].axis("off")

    plt.savefig("sample.png", dpi=700)

def main():
    root_dir = "/scratch/ramesh.anu/BDD/bdd100k/"
    bdd = BDD(root_dir)

if __name__ == "__main__":
    test()
    # main()
