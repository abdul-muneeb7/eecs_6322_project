
import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

checkpoint = config['DEFAULT']['checkpoint']
logdir = config['DEFAULT']['logdir']
pretrained_dir = config['DEFAULT']['pretrained_dir']
data_dir = config['DEFAULT']['data_dir']
json_list = config['DEFAULT']['json_list']
pretrained_model_name = config['DEFAULT']['pretrained_model_name']
save_checkpoint = config.getboolean('DEFAULT', 'save_checkpoint')
max_epochs = config.getint('DEFAULT', 'max_epochs')
batch_size = config.getint('DEFAULT', 'batch_size')
sw_batch_size = config.getint('DEFAULT', 'sw_batch_size')
optim_lr = config.getfloat('DEFAULT', 'optim_lr')
optim_name = config['DEFAULT']['optim_name']
reg_weight = config.getfloat('DEFAULT', 'reg_weight')
momentum = config.getfloat('DEFAULT', 'momentum')
noamp = config.getboolean('DEFAULT', 'noamp')
val_every = config.getint('DEFAULT', 'val_every')
distributed = config.getboolean('DEFAULT', 'distributed')
world_size = config.getint('DEFAULT', 'world_size')
rank = config.getint('DEFAULT', 'rank')
dist_url = config['DEFAULT']['dist-url']
dist_backend = config['DEFAULT']['dist-backend']
norm_name = config['DEFAULT']['norm_name']
workers = config.getint('DEFAULT', 'workers')
feature_size = config.getint('DEFAULT', 'feature_size')
in_channels = config.getint('DEFAULT', 'in_channels')
out_channels = config.getint('DEFAULT', 'out_channels')
use_normal_dataset = config.getboolean('DEFAULT', 'use_normal_dataset')
a_min = config.getfloat('DEFAULT', 'a_min')
a_max = config.getfloat('DEFAULT', 'a_max')
b_min = config.getfloat('DEFAULT', 'b_min')
b_max = config.getfloat('DEFAULT', 'b_max')
space_x = config.getfloat('DEFAULT', 'space_x')
space_y = config.getfloat('DEFAULT', 'space_y')
space_z = config.getfloat('DEFAULT', 'space_z')
roi_x = config.getint('DEFAULT', 'roi_x')
roi_y = config.getint('DEFAULT', 'roi_y')
roi_z = config.getint('DEFAULT', 'roi_z')
dropout_rate = config.getfloat('DEFAULT', 'dropout_rate')
dropout_path_rate = config.getfloat('DEFAULT', 'dropout_path_rate')
RandFlipd_prob = config.getfloat('DEFAULT', 'RandFlipd_prob')
RandRotate90d_prob = config.getfloat('DEFAULT', 'RandRotate90d_prob')
RandScaleIntensityd_prob = config.getfloat('DEFAULT', 'RandScaleIntensityd_prob')
RandShiftIntensityd_prob = config.getfloat('DEFAULT', 'RandShiftIntensityd_prob')
infer_overlap = config.getfloat('DEFAULT', 'infer_overlap')
lrschedule = config['DEFAULT']['lrschedule']
warmup_epochs = config.getint('DEFAULT', 'warmup_epochs')
resume_ckpt = config.getboolean('DEFAULT', 'resume_ckpt')
smooth_dr = config.getfloat('DEFAULT', 'smooth_dr')
smooth_nr = config.getfloat('DEFAULT', 'smooth_nr')

use_ssl_pretrained = config.getboolean('DEFAULT', 'use_ssl_pretrained')
use_checkpoint = config.getboolean('DEFAULT', 'use_checkpoint')
spatial_dims =config.getint('DEFAULT', 'spatial_dims')
squared_dice =config.getboolean('DEFAULT', 'squared_dice')

test_mode =config['DEFAULT'].getboolean('test_mode')

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader():
    data_dir = config['DEFAULT']['data_dir']
    datalist_json = os.path.join(data_dir, json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(space_x, space_y, space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(roi_x, roi_y, roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(space_x, space_y, space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys="image", pixdim=(space_x, space_y, space_z), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=workers
            )
        train_sampler = Sampler(train_ds) if distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
