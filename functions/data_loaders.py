import torch
import torchvision
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import warnings
import os
import numpy as np
from os.path import isfile, join

warnings.filterwarnings('ignore')


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


class NCaltech101(Dataset):
    def __init__(self, data_path='data/n-caltech/frames_number_10_split_by_number',
                 data_type='train', transform=False):

        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.dvs_filelist = []
        self.targets = []
        self.resize = transforms.Resize(size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        for i, cls in enumerate(self.clslist):
            # print (i, cls)
            file_list = os.listdir(os.path.join(self.filepath, cls))
            num_file = len(file_list)

            cut_idx = int(num_file * 0.9)
            train_file_list = file_list[:cut_idx]
            test_split_list = file_list[cut_idx:]
            for file in file_list:
                if data_type == 'train':
                    if file in train_file_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)
                else:
                    if file in test_split_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)

        self.data_num = len(self.dvs_filelist)
        self.data_type = data_type
        if data_type != 'train':
            counts = np.unique(np.array(self.targets), return_counts=True)[1]
            class_weights = counts.sum() / (counts * len(counts))
            self.class_weights = torch.Tensor(class_weights)
        self.classes = range(101)
        self.transform = transform
        self.rotate = transforms.RandomRotation(degrees=15)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-15, 15))

    def __getitem__(self, index):
        file_pth = self.dvs_filelist[index]
        label = self.targets[index]
        data = torch.from_numpy(np.load(file_pth)['frames']).float()
        data = self.resize(data)

        if self.transform:

            choices = ['roll', 'rotate', 'shear']
            aug = np.random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-3, 3)
                off2 = random.randint(-3, 3)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == 'rotate':
                data = self.rotate(data)
            if aug == 'shear':
                data = self.shearx(data)

        return data, label

    def __len__(self):
        return self.data_num


def build_ncaltech(transform=False):
    train_dataset = NCaltech101(transform=transform)
    val_dataset = NCaltech101(data_type='test', transform=False)

    return train_dataset, val_dataset


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.rotate = transforms.RandomRotation(degrees=30)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        data = self.resize(data.permute([3, 0, 1, 2]))

        if self.transform:

            choices = ['roll', 'rotate', 'shear']
            aug = np.random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == 'rotate':
                data = self.rotate(data)
            if aug == 'shear':
                data = self.shearx(data)

        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


def build_dvscifar(path='data/cifar-dvs', transform=False):
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = DVSCifar10(root=train_path, transform=transform)
    val_dataset = DVSCifar10(root=val_path, transform=False)

    return train_dataset, val_dataset


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[3]
    H = size[4]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(input, target, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()

    target_a = target
    target_b = target[rand_index]

    # generate mixed sample
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


if __name__ == '__main__':
    choices = ['roll', 'rotate', 'shear']
    aug = np.random.choice(choices)
    print(aug)


