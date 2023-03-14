import glob
import os

import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor

from lglutide import config


def creat_annotations(a_path, u_path):
    # read all the images in the folder
    a_images = glob.glob(a_path + "*.jpeg")
    u_images = glob.glob(u_path + "*.jpeg")

    print(f"Image in A: {len(a_images)}, Images in U: {len(u_images)}")

    # create the annotations file
    annotations = pd.DataFrame(columns=["image", "label"])

    # add the images and labels to the annotations file using frame.concat
    annotations = pd.concat(
        [annotations, pd.DataFrame({"image": a_images, "label": 1})]
    )
    annotations = pd.concat(
        [annotations, pd.DataFrame({"image": u_images, "label": 0})]
    )

    # shuffle the annotations file
    annotations = annotations.sample(frac=1).reset_index(drop=True)

    # divide the annotations in train.csv and test.csv
    train = annotations.iloc[: int(config.TRAIN_SIZE * len(annotations))]
    test = annotations.iloc[int(config.TRAIN_SIZE * len(annotations)) :]

    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    print("Annotations file created!")


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.image_path = self.img_labels.iloc[:, 0]
        self.img_labels = self.img_labels.iloc[:, 1]

        self.transform = transforms.ToTensor()

        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.image_path.iloc[idx]

        image = read_image(img_path)

        image = transforms.Resize((config.IMAGE_W, config.IMAGE_H))(image)

        label = self.img_labels.iloc[idx]

        # convert image to numpy array
        image = image.numpy()

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def make_annotations():
    # check if the annotations file exists
    if not os.path.exists("data/train.csv") or not os.path.exists("data/test.csv"):
        # if not, create the annotations file
        creat_annotations(a_path="data/A/", u_path="data/U/")
    else:
        print("Annotations file already exists!")


def create_dataloader(annotations):
    # create a custom dataset and dataloader
    custom_dataset = CustomImageDataset(annotations_file=annotations)

    dataloader = DataLoader(custom_dataset, batch_size=config.BATCHSIZE, shuffle=True)

    return dataloader


def make_data():
    # create the annotations file
    make_annotations()

    # create the dataloaders
    train = create_dataloader(annotations="data/train.csv")

    test = create_dataloader(annotations="data/test.csv")

    return train, test


# python main block
if __name__ == "__main__":
    make_data()
