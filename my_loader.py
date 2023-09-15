import argparse
import os, sys

from torch.utils.data.sampler import Sampler

sys.path.append('./')

import os.path as osp
# import torchvision

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import Cnetwork
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import random
from scipy.stats import dirichlet
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from data_load import mnist, svhn, usps
from PIL import ImageOps
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

SMAX = 10
MIXMATCH_FLAG = False


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = channel - mean
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
    """

    def __init__(self, mean=None, meanfile=None):
        if mean:
            self.mean = mean
        else:
            arr = np.load(meanfile)
            self.mean = torch.from_numpy(arr.astype('float32') / 255.0)[[2, 1, 0], :, :]

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m in zip(tensor, self.mean):
            t.sub_(m)
        return tensor


def get_transform(dataset, img_size):
    if dataset in ['svhn2mnist', 'usps2mnist', 'mnist2usps']:
        transform_source = transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        transform_target = transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    elif dataset in ['visda17', 'office-home']:
        transform_source = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
        ])
        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
        ])
    else:
        transform_source = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
        ])

        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
        ])

    return transform_source, transform_source, transform_test


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size + 32, resize_size + 32)),
        transforms.RandomCrop(resize_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
    ])


def image_train_224(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size + 32, resize_size + 32), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(resize_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Normalize(meanfile='./data/ilsvrc_2012_mean.npy'),
        transforms.RandomCrop(crop_size),
    ])


def image_test_224(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        Normalize(meanfile='./data/ilsvrc_2012_mean.npy'),
        transforms.CenterCrop(crop_size),
    ])


def image_target(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        # transforms.CenterCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
    ])


def digit_train(resize_size=32):
    return transforms.Compose([
        transforms.Resize(32),
        # transforms.Lambda(lambda x: x.convert("RGB")),
        # transforms.RandomCrop(25),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def digit_test(resize_size=32):
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0],
                       np.array([int(la) for la in val.split()[1:]]))
                      for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1]))
                      for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageList_idx(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


class ImageList_twice(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        if self.transform is not None:
            img_raw = self.loader(path)
            img_0 = self.transform(img_raw)
            img_raw = self.loader(path)
            img_1 = self.transform(img_raw)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_0, img_1, target, index

    def __len__(self):
        return len(self.imgs)


def label_squeezing_collate_fn(batch):
    x, y, indx = default_collate(batch)
    return x, y.long().squeeze(), indx


def label_squeezing_collate_fn_twice(batch):
    x1, x2, y, indx = default_collate(batch)
    return x1, x2, y.long().squeeze(), indx


def _colorize_grayscale_image(image):
    return ImageOps.colorize(image, (0, 0, 0), (255, 255, 255))


def CL_load(args, transform=None):
    task_order = {}
    # task_order['office-home'] = ['A', 'C', 'P', 'R']
    task_order['office31'] = ['a', 'd', 'w']
    nClasses = {'office31': 31, 'office-home': 65}
    dset_loaders = {}
    prep_dict = {}
    if transform == None:
        if args.img_size == 224:
            prep_dict['train'] = image_train_224()
        else:
            prep_dict['train'] = image_train()
    else:
        prep_dict['train'] = transform
    if args.img_size == 224:
        prep_dict['test'] = image_test_224()
    else:
        prep_dict['test'] = image_test()
    idx = 0
    task_nClasses = []
    for data in task_order.keys():
        task = task_order[data]
        for t in task:
            task_nClasses.append((idx, nClasses[data]))
            tr = './data/{}/{}_list.txt'.format(data, t)
            txt_src = open(tr).readlines()
            dsize = len(txt_src)
            tv_size = int(0.8 * dsize)
            tr, ts = torch.utils.data.random_split(txt_src, [tv_size, dsize - tv_size])
            train = ImageList_idx(tr, transform=prep_dict['train'])
            test = ImageList_idx(ts, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True)
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size // 2, shuffle=False, num_workers=args.worker,
                                                       drop_last=False)
            dset_loaders[str(idx) + 'nClasses'] = nClasses[data]
            idx += 1
    return dset_loaders, task_nClasses


def office31_load(args):
    task_order = args.order.split('2')
    task = [i + '_list' for i in task_order]
    dset_loaders = {}
    prep_dict = {}
    prep_dict['train'] = image_train()
    prep_dict['test'] = image_test()
    for idx, t in enumerate(task):
        tr = './data/office31/{}.txt'.format(t)
        txt_src = open(tr).readlines()
        if idx == 0:
            dsize = len(txt_src)
            tv_size = int(0.8 * dsize)
            tr, ts = torch.utils.data.random_split(txt_src, [tv_size, dsize - tv_size])
            train = ImageList_idx(tr, transform=prep_dict['train'])
            test = ImageList_idx(ts, transform=prep_dict['test'])
            train_twice = ImageList_twice(txt_src, transform=prep_dict['train'])
            train_cvae = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size * args.grad_iter, shuffle=True, num_workers=args.worker,
                                                       drop_last=False)
            dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=args.batch_size * args.grad_iter, shuffle=True,
                                                             num_workers=args.worker,
                                                             drop_last=False)
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,
                                                       drop_last=False)
        else:
            train = ImageList_idx(txt_src, transform=prep_dict['train'])
            train_twice = ImageList_twice(txt_src, transform=prep_dict['train'])
            train_cvae = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                             drop_last=False)
            dset_loaders[str(idx) + 'tr_twice_gen'] = DataLoader(train_twice, batch_size=args.batch_size * args.grad_iter, shuffle=True,
                                                                 num_workers=args.worker,
                                                                 drop_last=False)
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
            num_sample = len(dset_loaders[str(idx) + 'tr'].dataset)
            label_bank_raw = torch.zeros(num_sample, dtype=torch.int64)
            iter_test = iter(dset_loaders[str(idx) + 'tr'])
            for i in range(len(dset_loaders[str(idx) + 'tr'])):
                data = iter_test.next()
                labels = data[1]
                indx = data[-1]
                label_bank_raw[indx] = labels
            if args.shot > 0:
                few_short_index = []
                for c in range(args.class_num):
                    index_c = np.where(label_bank_raw == c)[0]
                    few_short_index.extend(np.random.choice(index_c, args.shot, replace=False))
                few_short_index = np.array(few_short_index)
                few_short_index = np.sort(few_short_index)
                dset_loaders[str(idx) + 'few_short_index'] = few_short_index
                unlabel_index = [i for i in range(len(txt_src)) if i not in few_short_index]
                dset_loaders[str(idx) + 'unlabel_index'] = np.array(unlabel_index)
                ts = [txt_src[i] for i in range(len(txt_src)) if i not in few_short_index]
                labeled = [txt_src[i] for i in range(len(txt_src)) if i in few_short_index]
                labeled_data = ImageList_idx(labeled, transform=prep_dict['train'])
                unlabeled_data = ImageList_twice(ts, transform=prep_dict['train'])
                test = ImageList_idx(ts, transform=prep_dict['test'])
                dset_loaders[str(idx) + 'tr_labeled'] = DataLoader(labeled_data, batch_size=args.batch_size, shuffle=True,
                                                                   num_workers=args.worker, drop_last=True)
                dset_loaders[str(idx) + 'tr_twice_unlabeled'] = DataLoader(unlabeled_data, batch_size=args.batch_size, shuffle=True,
                                                                           num_workers=args.worker,
                                                                           drop_last=True)
            else:
                test = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,
                                                       drop_last=False)
        print('Loaded data of task {}'.format(idx))
    return dset_loaders


def office31_load_BBF(args):
    task_order = args.order.split('2')
    task = [i + '_list' for i in task_order]
    dset_loaders = {}
    prep_dict = {}
    prep_dict['train'] = image_train()
    prep_dict['test'] = image_test()
    t = task[0]
    print(t)
    for idx in range(10):
        tr = './data/office31/{}.txt'.format(t)
        txt_src = open(tr).readlines()

        dsize = len(txt_src)
        tv_size = int(0.67 * dsize)
        tr, ts = torch.utils.data.random_split(txt_src, [tv_size, dsize - tv_size])
        train = ImageList_idx(tr, transform=prep_dict['train'])
        test = ImageList_idx(ts, transform=prep_dict['test'])
        train2 = ImageList_idx(tr, transform=prep_dict['test'])
        dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size * args.grad_iter, shuffle=True, num_workers=args.worker,
                                                   drop_last=False)
        dset_loaders[str(idx) + 'tr2'] = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
        dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
    return dset_loaders


def officeHome_load_BBF(args):
    task_order = args.order.split('2')
    task = [i + '_list' for i in task_order]
    dset_loaders = {}
    prep_dict = {}
    prep_dict['train'] = image_train()
    prep_dict['test'] = image_test()
    t = task[0]
    print(t)
    for idx in range(10):
        tr = './data/office-home/{}.txt'.format(t)
        txt_src = open(tr).readlines()

        dsize = len(txt_src)
        tv_size = int(0.67 * dsize)
        tr, ts = torch.utils.data.random_split(txt_src, [tv_size, dsize - tv_size])
        train = ImageList_idx(tr, transform=prep_dict['train'])
        test = ImageList_idx(ts, transform=prep_dict['test'])
        train2 = ImageList_idx(tr, transform=prep_dict['test'])
        dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size * args.grad_iter, shuffle=True, num_workers=args.worker,
                                                   drop_last=False)
        dset_loaders[str(idx) + 'tr2'] = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
        dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
    return dset_loaders


def officeHome_load(args):
    task_order = args.order.split('2')
    task = [i + '_list' for i in task_order]
    dset_loaders = {}
    prep_dict = {}
    prep_dict['train'] = image_train()
    prep_dict['test'] = image_test()
    # label_num = {'a':45,'d':10,'w':24}

    for idx, t in enumerate(task):
        tr = './data/office-home/{}.txt'.format(t)
        txt_src = open(tr).readlines()
        if idx == 0:
            dsize = len(txt_src)
            tv_size = int(0.8 * dsize)
            tr, ts = torch.utils.data.random_split(txt_src, [tv_size, dsize - tv_size])
            train = ImageList_idx(tr, transform=prep_dict['train'])
            test = ImageList_idx(ts, transform=prep_dict['test'])
            train_twice = ImageList_twice(txt_src, transform=prep_dict['train'])
            train_cvae = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size * args.grad_iter, shuffle=True, num_workers=args.worker,
                                                       drop_last=False)
            dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                             drop_last=False)
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                       drop_last=False)
        else:
            train = ImageList_idx(txt_src, transform=prep_dict['train'])
            train_twice = ImageList_twice(txt_src, transform=prep_dict['train'])
            train_cvae = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                             drop_last=False)
            dset_loaders[str(idx) + 'tr_twice_gen'] = DataLoader(train_twice, batch_size=args.batch_size * args.grad_iter, shuffle=True,
                                                                 num_workers=args.worker,
                                                                 drop_last=False)
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
            num_sample = len(dset_loaders[str(idx) + 'tr'].dataset)
            label_bank_raw = torch.zeros(num_sample, dtype=torch.int64)
            iter_test = iter(dset_loaders[str(idx) + 'tr'])
            for i in range(len(dset_loaders[str(idx) + 'tr'])):
                data = iter_test.next()
                labels = data[1]
                indx = data[-1]
                label_bank_raw[indx] = labels
            if args.shot > 0:
                few_short_index = []
                for c in range(args.class_num):
                    index_c = np.where(label_bank_raw == c)[0]
                    few_short_index.extend(np.random.choice(index_c, args.shot, replace=False))
                few_short_index = np.array(few_short_index)
                few_short_index = np.sort(few_short_index)
                dset_loaders[str(idx) + 'few_short_index'] = few_short_index
                unlabel_index = [i for i in range(len(txt_src)) if i not in few_short_index]
                dset_loaders[str(idx) + 'unlabel_index'] = np.array(unlabel_index)
                ts = [txt_src[i] for i in range(len(txt_src)) if i not in few_short_index]
                labeled = [txt_src[i] for i in range(len(txt_src)) if i in few_short_index]
                labeled_data = ImageList_idx(labeled, transform=prep_dict['train'])
                unlabeled_data = ImageList_twice(ts, transform=prep_dict['train'])
                test = ImageList_idx(ts, transform=prep_dict['test'])
                dset_loaders[str(idx) + 'tr_labeled'] = DataLoader(labeled_data, batch_size=args.batch_size, shuffle=True,
                                                                   num_workers=args.worker, drop_last=True)
                dset_loaders[str(idx) + 'tr_twice_unlabeled'] = DataLoader(unlabeled_data, batch_size=args.batch_size, shuffle=True,
                                                                           num_workers=args.worker,
                                                                           drop_last=True)
            else:
                test = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                       drop_last=False)
        print('Loaded data of task {}'.format(idx))
    return dset_loaders


def visda_load(args):
    task_order = args.order.split('2')
    task = [i + '_list' for i in task_order]
    dset_loaders = {}
    prep_dict = {}
    prep_dict['train'] = image_train()
    prep_dict['test'] = image_test()
    for idx, t in enumerate(task):
        tr = './data/visda/{}.txt'.format(t)
        txt_src = open(tr).readlines()
        if idx == 0:
            dsize = len(txt_src)
            tv_size = int(0.9 * dsize)
            tr, ts = torch.utils.data.random_split(txt_src, [tv_size, dsize - tv_size])
            train = ImageList_idx(tr, transform=prep_dict['train'])
            test = ImageList_idx(ts, transform=prep_dict['test'])
            train_twice = ImageList_twice(txt_src, transform=prep_dict['train'])
            train_cvae = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                             drop_last=False)
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,
                                                       drop_last=False)
        else:
            train = ImageList_idx(txt_src, transform=prep_dict['train'])
            train_twice = ImageList_twice(txt_src, transform=prep_dict['train'])
            train_cvae = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                             drop_last=False)
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
            num_sample = len(dset_loaders[str(idx) + 'tr'].dataset)
            label_bank_raw = torch.zeros(num_sample, dtype=torch.int64)
            iter_test = iter(dset_loaders[str(idx) + 'tr'])
            for i in range(len(dset_loaders[str(idx) + 'tr'])):
                data = iter_test.next()
                labels = data[1]
                indx = data[-1]
                label_bank_raw[indx] = labels
            if args.shot > 0:
                few_short_index = []
                for c in range(args.class_num):
                    index_c = np.where(label_bank_raw == c)[0]
                    few_short_index.extend(np.random.choice(index_c, args.shot, replace=False))
                few_short_index = np.array(few_short_index)
                few_short_index = np.sort(few_short_index)
                dset_loaders[str(idx) + 'few_short_index'] = few_short_index
                unlabel_index = [i for i in range(len(txt_src)) if i not in few_short_index]
                dset_loaders[str(idx) + 'unlabel_index'] = np.array(unlabel_index)
                ts = [txt_src[i] for i in range(len(txt_src)) if i not in few_short_index]
                labeled = [txt_src[i] for i in range(len(txt_src)) if i in few_short_index]
                labeled_data = ImageList_idx(labeled, transform=prep_dict['train'])
                unlabeled_data = ImageList_twice(ts, transform=prep_dict['train'])
                test = ImageList_idx(ts, transform=prep_dict['test'])
                dset_loaders[str(idx) + 'tr_labeled'] = DataLoader(labeled_data, batch_size=args.batch_size, shuffle=True,
                                                                   num_workers=args.worker, drop_last=False)
                dset_loaders[str(idx) + 'tr_twice_unlabeled'] = DataLoader(unlabeled_data, batch_size=args.batch_size, shuffle=True,
                                                                           num_workers=args.worker,
                                                                           drop_last=True)
            else:
                test = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,
                                                       drop_last=False)
        print('Loaded data of task {}'.format(idx))
    return dset_loaders


def digit_load(args):
    task_order = args.order.split('2')
    task_bank = {'m': 'mnist', 's': 'svhn', 'u': 'usps'}
    task = [task_bank[i] for i in task_order]
    dset_loaders = {}
    prep_dict = {}
    prep_dict['train'] = digit_train()
    prep_dict['test'] = digit_test()
    # label_num = {'a':45,'d':10,'w':24}
    for idx, t in enumerate(task):
        tr = './data/{}/train.txt'.format(t)
        ts = './data/{}/test.txt'.format(t)
        txt_train = open(tr).readlines()
        txt_test = open(ts).readlines()
        if idx == 0:
            # dsize = len(txt_src)
            # tv_size = int(0.8 * dsize)
            # tr, ts = torch.utils.data.random_split(txt_src, [tv_size, dsize - tv_size])
            train = ImageList_idx(txt_train, transform=prep_dict['train'])
            test = ImageList_idx(txt_test, transform=prep_dict['test'])
            train_twice = ImageList_twice(txt_train, transform=prep_dict['train'])
            train_cvae = ImageList_idx(txt_train, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                             drop_last=False)
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,
                                                       drop_last=False)
        else:
            train = ImageList_idx(txt_train, transform=prep_dict['train'])
            test = ImageList_idx(txt_test, transform=prep_dict['test'])
            train_twice = ImageList_twice(txt_train, transform=prep_dict['train'])
            train_cvae = ImageList_idx(txt_train, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                             drop_last=False)
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,
                                                       drop_last=False)
            num_sample = len(dset_loaders[str(idx) + 'tr'].dataset)
            label_bank_raw = torch.zeros(num_sample, dtype=torch.int64)
            iter_test = iter(dset_loaders[str(idx) + 'tr'])
            for i in range(len(dset_loaders[str(idx) + 'tr'])):
                data = iter_test.next()
                labels = data[1]
                indx = data[-1]
                label_bank_raw[indx] = labels
            if args.shot > 0:
                few_short_index = []
                for c in range(args.class_num):
                    index_c = np.where(label_bank_raw == c)[0]
                    few_short_index.extend(np.random.choice(index_c, args.shot, replace=False))
                few_short_index = np.array(few_short_index)
                few_short_index = np.sort(few_short_index)
                dset_loaders[str(idx) + 'few_short_index'] = few_short_index
                unlabel_index = [i for i in range(len(txt_train)) if i not in few_short_index]
                dset_loaders[str(idx) + 'unlabel_index'] = np.array(unlabel_index)
                unlabel = [txt_train[i] for i in range(len(txt_train)) if i not in few_short_index]
                labeled = [txt_train[i] for i in range(len(txt_train)) if i in few_short_index]
                labeled_data = ImageList_idx(labeled, transform=prep_dict['train'])
                unlabeled_data = ImageList_twice(unlabel, transform=prep_dict['train'])
                # test = ImageList_idx(ts, transform=prep_dict['test'])
                dset_loaders[str(idx) + 'tr_labeled'] = DataLoader(labeled_data, batch_size=args.batch_size, shuffle=True,
                                                                   num_workers=args.worker, drop_last=True)
                dset_loaders[str(idx) + 'tr_twice_unlabeled'] = DataLoader(unlabeled_data, batch_size=args.batch_size, shuffle=True,
                                                                           num_workers=args.worker,
                                                                           drop_last=True)
        print('Loaded data of task {}'.format(idx))
    return dset_loaders


def officeHome1_load(args):
    task_order = args.order.split('2')
    task = [i + '_list' for i in task_order]
    dset_loaders = {}
    prep_dict = {}
    prep_dict['train'] = image_train()
    prep_dict['test'] = image_test()
    for idx, t in enumerate(task):
        tr = './data/office-home/{}.txt'.format(t)
        txt_src = open(tr).readlines()
        if idx == 0:
            dsize = len(txt_src)
            tv_size = int(0.8 * dsize)
            tr, ts = torch.utils.data.random_split(txt_src, [tv_size, dsize - tv_size])
            train = ImageList_idx(tr, transform=prep_dict['train'])
            test = ImageList_idx(ts, transform=prep_dict['test'])
            train_twice = ImageList_twice(txt_src, transform=prep_dict['train'])
            train_cvae = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                             drop_last=False)
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,
                                                       drop_last=False)
        else:
            train = ImageList_idx(txt_src, transform=prep_dict['train'])
            train_twice = ImageList_twice(txt_src, transform=prep_dict['train'])
            train_cvae = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                             drop_last=False)
            dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
            num_sample = len(dset_loaders[str(idx) + 'tr'].dataset)
            label_bank_raw = torch.zeros(num_sample, dtype=torch.int64)
            iter_test = iter(dset_loaders[str(idx) + 'tr'])
            for i in range(len(dset_loaders[str(idx) + 'tr'])):
                data = iter_test.next()
                labels = data[1]
                indx = data[-1]
                label_bank_raw[indx] = labels
            if args.shot > 0:
                few_short_index = []
                for c in range(args.class_num):
                    index_c = np.where(label_bank_raw == c)[0]
                    few_short_index.extend(np.random.choice(index_c, args.shot, replace=False))
                few_short_index = np.array(few_short_index)
                # few_short_index = np.sort(few_short_index)
                dset_loaders[str(idx) + 'few_short_index'] = few_short_index
                ts = [txt_src[i] for i in range(len(txt_src)) if i not in few_short_index]
                test = ImageList_idx(ts, transform=prep_dict['test'])
            else:
                test = ImageList_idx(txt_src, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
            dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,
                                                       drop_last=False)
        print('Loaded data of task {}'.format(idx))
    return dset_loaders


def digit0_load(args):
    train_bs = args.batch_size
    task_order = args.order.split('2')
    task = [i for i in task_order]
    dset_loaders = {}
    for idx, t in enumerate(task):
        if t == 'm':
            data = 'mnist'
            train = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(32),
                                        transforms.Lambda(lambda x: x.convert("RGB")),
                                        # transforms.RandomCrop(25),
                                        transforms.RandomRotation(10),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))
            train_twice = mnist.MNIST_twice('./data/mnist/', train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(32),
                                                transforms.Lambda(lambda x: x.convert("RGB")),
                                                transforms.RandomCrop(25),
                                                transforms.RandomRotation(10),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))
            test = mnist.MNIST_idx('./data/mnist/', train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(32),
                                       transforms.Lambda(lambda x: x.convert("RGB")),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))
            train_cvae = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(32),
                                             transforms.Lambda(lambda x: x.convert("RGB")),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))
        elif t == 's':
            data = 'svhn'
            train = svhn.SVHN_idx('./data/svhn/', split='train', download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize(32),
                                      transforms.RandomCrop(25),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]))
            train_twice = svhn.SVHN_twice('./data/svhn/', split='train', download=True,
                                          transform=transforms.Compose([
                                              transforms.Resize(32),
                                              transforms.RandomCrop(25),
                                              transforms.RandomRotation(10),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ]))
            test = svhn.SVHN_idx('./data/svhn/', split='test', download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(32),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]))
            train_cvae = svhn.SVHN_idx('./data/svhn/', split='train', download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(32),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))
        elif t == 'u':
            data = 'usps'
            train = usps.USPS_idx('./data/usps/', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize(32),
                                      transforms.Lambda(lambda x: x.convert("RGB")),
                                      transforms.RandomCrop(25),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]))
            train_twice = usps.USPS_twice('./data/usps/', train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.Resize(32),
                                              transforms.Lambda(lambda x: x.convert("RGB")),
                                              transforms.RandomCrop(25),
                                              transforms.RandomRotation(10),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ]))
            test = usps.USPS_idx('./data/usps/', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(32),
                                     transforms.Lambda(lambda x: x.convert("RGB")),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]))
            train_cvae = usps.USPS_idx('./data/usps/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(32),
                                           transforms.Lambda(lambda x: x.convert("RGB")),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))

        dset_loaders[str(idx) + 'tr'] = DataLoader(train, batch_size=train_bs, shuffle=True,
                                                   collate_fn=(label_squeezing_collate_fn or default_collate),
                                                   **({'num_workers': args.worker, 'pin_memory': True}), drop_last=False)
        dset_loaders[str(idx) + 'tr_twice'] = DataLoader(train_twice, batch_size=train_bs, shuffle=True,
                                                         collate_fn=(label_squeezing_collate_fn_twice or default_collate),
                                                         **({'num_workers': args.worker, 'pin_memory': True}), drop_last=False)
        dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=train_bs, shuffle=False,
                                                   collate_fn=(label_squeezing_collate_fn or default_collate),
                                                   **({'num_workers': args.worker, 'pin_memory': True}), drop_last=False)
        dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=train_bs, shuffle=False,
                                                   collate_fn=(label_squeezing_collate_fn or default_collate),
                                                   **({'num_workers': args.worker, 'pin_memory': True}), drop_last=False)
        num_sample = len(dset_loaders[str(idx) + 'tr'].dataset)
        label_bank_raw = torch.zeros(num_sample, dtype=torch.int64)
        iter_test = iter(dset_loaders[str(idx) + 'tr'])
        for i in range(len(dset_loaders[str(idx) + 'tr'])):
            data = iter_test.next()
            labels = data[1]
            indx = data[-1]
            label_bank_raw[indx] = labels
        if args.shot > 0:
            few_short_index = []
            for c in range(args.class_num):
                index_c = np.where(label_bank_raw == c)[0]
                few_short_index.extend(np.random.choice(index_c, args.shot, replace=False))
            few_short_index = np.array(few_short_index)
            few_short_index = np.sort(few_short_index)
            dset_loaders[str(idx) + 'few_short_index'] = few_short_index
            unlabel_index = [i for i in range(num_sample) if i not in few_short_index]
            dset_loaders[str(idx) + 'unlabel_index'] = np.array(unlabel_index)
            # ts = [txt_src[i] for i in range(len(txt_src)) if i not in few_short_index]
            labeled = [txt_src[i] for i in range(len(txt_src)) if i in few_short_index]
            labeled_data = ImageList_idx(labeled, transform=prep_dict['train'])
            unlabeled_data = ImageList_twice(ts, transform=prep_dict['train'])
            test = ImageList_idx(ts, transform=prep_dict['test'])
            dset_loaders[str(idx) + 'tr_labeled'] = DataLoader(labeled_data, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,
                                                               drop_last=True)
            dset_loaders[str(idx) + 'tr_twice_unlabeled'] = DataLoader(unlabeled_data, batch_size=args.batch_size, shuffle=True,
                                                                       num_workers=args.worker,
                                                                       drop_last=True)
        else:
            test = ImageList_idx(txt_src, transform=prep_dict['test'])
        dset_loaders[str(idx) + 'ts'] = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
        dset_loaders[str(idx) + 'gr'] = DataLoader(train_cvae, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,
                                                   drop_last=False)
        print('task {} data {} loaded'.format(idx, data))
    return dset_loaders


def ten_class_load(args):
    train_bs = args.batch_size
    task_order = args.order.split('2')
    task = [i for i in task_order]
    dset_loaders = {}
    for idx, t in enumerate(task):
        if t == 'm':
            data = 'mnist'
            train_source = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.ToPILImage(),
                                               transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
                                               # transforms.RandomHorizontalFlip(),
                                               transforms.Pad(2),
                                               transforms.ToTensor(),
                                           ]))
            sample_source = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.ToPILImage(),
                                                transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
                                                # transforms.RandomHorizontalFlip(),
                                                transforms.Pad(2),
                                                transforms.ToTensor(),
                                            ]))
            test_source = mnist.MNIST_idx('./data/mnist/', train=False, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
                                              transforms.Pad(2),
                                              transforms.ToTensor(),
                                          ]))
        elif t == 's':
            data = 'svhn'
            train_source = svhn.SVHN_idx('./data/svhn/', split='train', download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(28),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.Pad(2),
                                             transforms.ToTensor(),
                                             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))
            sample_source = svhn.SVHN_idx('./data/svhn/', split='train', download=True,
                                          transform=transforms.Compose([
                                              transforms.Resize(28),
                                              # transforms.RandomHorizontalFlip(),
                                              transforms.Pad(2),
                                              transforms.ToTensor(),
                                              # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ]))
            test_source = svhn.SVHN_idx('./data/svhn/', split='test', download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(28),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.Pad(2),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
        elif t == 'u':
            data = 'usps'
            train_source = usps.USPS_idx('./data/usps/', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(28),
                                             transforms.ToTensor(),
                                             transforms.ToPILImage(),
                                             transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.Pad(2),
                                             transforms.ToTensor(),
                                         ]))
            sample_source = usps.USPS_idx('./data/usps/', train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.Resize(28),
                                              transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
                                              # transforms.RandomHorizontalFlip(),
                                              transforms.Pad(2),
                                              transforms.ToTensor(),
                                          ]))
            test_source = usps.USPS_idx('./data/usps/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                            transforms.ToPILImage(),
                                            transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
                                            transforms.Pad(2),
                                            transforms.ToTensor(),
                                        ]))

        dset_loaders[str(idx) + 'tr'] = DataLoader(train_source, batch_size=train_bs, shuffle=True,
                                                   collate_fn=(label_squeezing_collate_fn or default_collate),
                                                   **({'num_workers': args.worker, 'pin_memory': True}), drop_last=False)
        dset_loaders[str(idx) + 'gr'] = DataLoader(sample_source, batch_size=train_bs, shuffle=True,
                                                   collate_fn=(label_squeezing_collate_fn or default_collate),
                                                   **({'num_workers': args.worker, 'pin_memory': True}), drop_last=False)
        dset_loaders[str(idx) + 'ts'] = DataLoader(test_source, batch_size=train_bs, shuffle=False,
                                                   collate_fn=(label_squeezing_collate_fn or default_collate),
                                                   **({'num_workers': args.worker, 'pin_memory': True}), drop_last=False)
        print('task {} data {} loaded'.format(idx, data))
    return dset_loaders
