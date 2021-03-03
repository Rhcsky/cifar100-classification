import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# import albumentations as A
# from albumentations.pytorch import ToTensorV2


def get_dataloader(args):
    train_dir = './data/train_loader'
    test_dir = './data/test_loader'

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print("Data already exist.")
        train_loader = torch.load(train_dir)
        test_loader = torch.load(test_dir)
        return train_loader, test_loader

    normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # transform_albumentations = A.Compose([
    #     A.Resize(36, 36),
    #     A.RandomCrop(32, 32),
    #     A.OneOf([
    #         A.HorizontalFlip(p=1),
    #         A.RandomRotate90(p=1),
    #         A.VerticalFlip(p=1)
    #     ], p=1),
    #     A.OneOf([
    #         A.MotionBlur(p=1),
    #         A.OpticalDistortion(p=1),
    #         A.GaussNoise(p=1)
    #     ], p=1),
    #     normalize,
    #     ToTensorV2(),
    # ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(datasets.CIFAR100('./data', train=False, transform=transform_test),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    torch.save(train_loader, train_dir)
    torch.save(test_loader, test_dir)
    print("Save train loader and test loader in './data/'")

    return train_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label
