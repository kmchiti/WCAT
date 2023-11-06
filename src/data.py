import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np
import logging
from common import FromParams, Registrable, Params, Lazy


class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target,


class Dataset(Registrable):
    def __init__(self, _data_root: str = './data', batch_size: int = 256, _num_workers: int = 1, augment: bool = True,
                 attack_sample_size: int = None, drop_last: bool = True, pin_memory: bool = True, shuffle: bool = True):
        self.data_root = _data_root
        self.batch_size = batch_size
        self.num_workers = _num_workers
        self.augment = augment
        self.attack_sample_size = attack_sample_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.shuffle = shuffle

    def _creat_data_transform(self, split: str = 'train'):
        raise NotImplementedError()

    def _creat_dataset(self, split: str = 'train'):
        raise NotImplementedError()

    @staticmethod
    def _creat_sampler(dataset: dset, sample_size: int, shuffle: bool = True):
        indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(indices)
        major_idx, minor_idx = indices[sample_size:], indices[:sample_size]
        major_sampler = SubsetRandomSampler(major_idx)
        minor_sampler = SubsetRandomSampler(minor_idx)
        return major_sampler, minor_sampler

    def build(self):
        raise NotImplementedError()


@Dataset.register("imagenet")
class _ImageNet(Dataset):
    def __init__(self, _ILSVRC2012_img_path: str = '/network/datasets/imagenet',
                 _imagenet_val_script: str = './src/imagenet_val.sh',
                 _use_train: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.ILSVRC2012_img_path = _ILSVRC2012_img_path
        self.imagenet_val_script = _imagenet_val_script
        self.use_train = _use_train
        self.num_classes = 1000

    def _copy_dataset2slurm_dir(self):

        if "SLURM_JOB_ID" in os.environ.keys():
            ILSVRC2012_img_train_path = os.path.join(self.ILSVRC2012_img_path, 'ILSVRC2012_img_train.tar')
            ILSVRC2012_img_val_path = os.path.join(self.ILSVRC2012_img_path, 'ILSVRC2012_img_val.tar')
            assert os.path.exists(ILSVRC2012_img_train_path)
            assert os.path.exists(ILSVRC2012_img_val_path)
            assert os.path.exists(self.imagenet_val_script)
            dataset_home = os.path.join(os.environ["SLURM_TMPDIR"], "ImageNet")
            if not os.path.exists(os.path.join(dataset_home)):
                os.system(f"mkdir {dataset_home} > /dev/null")

            if self.use_train:
                if os.path.exists(os.path.join(dataset_home, "train")):
                    logging.info(f"ImageNet train data found in {dataset_home}/train")
                else:
                    os.system(f"mkdir {dataset_home}/train > /dev/null")
                    logging.info("starting to copy train data to SLURM_TMPDIR")
                    os.system(f"tar -xvf {ILSVRC2012_img_train_path} -C {dataset_home}/train/ > /dev/null")
                    os.system(
                        'find ' + dataset_home + '/train/ -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done > /dev/null')
            if os.path.exists(os.path.join(dataset_home, "val")):
                logging.info(f"ImageNet val data found in {dataset_home}/val")
            else:
                os.system(f"mkdir {dataset_home}/val > /dev/null")
                logging.info("starting to copy val data to SLURM_TMPDIR")
                os.system(f"tar -xvf {ILSVRC2012_img_val_path} -C {dataset_home}/val/ > /dev/null")
                os.chmod(self.imagenet_val_script, os.stat(self.imagenet_val_script).st_mode | 0o100)
                os.system(f"{self.imagenet_val_script} {dataset_home}/val/ > /dev/null")
        else:
            logging.warning("use ILSVRC2012_img_path as dataset home")
            dataset_home = self.ILSVRC2012_img_path
        return dataset_home

    def _creat_data_transform(self, split: str = 'train'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.augment and split == 'train':
            data_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            data_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        return data_transform

    def _creat_dataset(self, split: str = 'train'):
        if split not in ['train', 'val']:
            raise ValueError("Invalid 'split' parameter. It must be 'train' or 'val'.")
        dataset_home = self._copy_dataset2slurm_dir()
        data_dir = os.path.join(dataset_home, split)
        dataset = ImageNet(data_dir, self._creat_data_transform(split))
        return dataset

    def build(self):

        if self.use_train:
            train_loader = torch.utils.data.DataLoader(self._creat_dataset('train'), batch_size=self.batch_size,
                                                       shuffle=self.shuffle, num_workers=self.num_workers,
                                                       pin_memory=self.pin_memory, drop_last=self.drop_last)
        else:
            train_loader = None
        validation_dataset = self._creat_dataset('val')
        if self.attack_sample_size is None:
            val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size,
                                                     shuffle=False, num_workers=self.num_workers,
                                                     pin_memory=self.pin_memory)
            calibration_loader = None
        else:
            valid_sampler, calibration_sampler = self._creat_sampler(validation_dataset, self.attack_sample_size)
            val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size,
                                                     shuffle=False, num_workers=self.num_workers,
                                                     pin_memory=self.pin_memory, sampler=valid_sampler)
            calibration_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.attack_sample_size,
                                                             shuffle=False, num_workers=self.num_workers,
                                                             pin_memory=self.pin_memory, sampler=calibration_sampler)

        return train_loader, val_loader, val_loader, calibration_loader


@Dataset.register("cifar10")
class _CIFAR10(Dataset):
    def __init__(self, valid_size: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.valid_size = valid_size
        self.num_classes = 10

    def _creat_data_transform(self, split: str = 'train'):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        if self.augment and split == 'train':
            data_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        return data_transform

    def _creat_dataset(self, split: str = 'train'):
        if split not in ['train', 'val', 'test']:
            raise ValueError("Invalid 'split' parameter. It must be 'train' or 'val'.")
        dataset = dset.CIFAR10(self.data_root, train=(split in ['train', 'val']), download=True,
                               transform=self._creat_data_transform(split))
        return dataset

    def build(self):

        train_dataset = self._creat_dataset(split='train')
        val_dataset = self._creat_dataset(split='val')
        test_dataset = self._creat_dataset(split='test')
        train_sampler, valid_sampler = self._creat_sampler(train_dataset, int(np.floor(self.valid_size * len(train_dataset))), self.shuffle)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                                   shuffle=False, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory, drop_last=self.drop_last)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                                   shuffle=False, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory, drop_last=self.drop_last)

        if self.attack_sample_size is None:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory)
            calibration_loader = None
        else:
            test_sampler, calibration_sampler = self._creat_sampler(test_dataset, self.attack_sample_size)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory, sampler=test_sampler)
            calibration_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.attack_sample_size,
                                                             shuffle=False, num_workers=self.num_workers,
                                                             pin_memory=self.pin_memory, sampler=calibration_sampler)

        return train_loader, test_loader, valid_loader, calibration_loader


@Dataset.register("mnist")
class _MNIST(Dataset):
    def __init__(self, valid_size: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.valid_size = valid_size
        self.num_classes = 10

    def _creat_data_transform(self, split: str = 'train'):
        normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        if self.augment and split == 'train':
            data_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        return data_transform

    def _creat_dataset(self, split: str = 'train'):
        if split not in ['train', 'val', 'test']:
            raise ValueError("Invalid 'split' parameter. It must be 'train' or 'val'.")
        dataset = dset.MNIST(self.data_root, train=(split in ['train', 'val']), download=True,
                             transform=self._creat_data_transform(split))
        return dataset

    def build(self):

        train_dataset = self._creat_dataset(split='train')
        val_dataset = self._creat_dataset(split='val')
        test_dataset = self._creat_dataset(split='test')
        train_sampler, valid_sampler = self._creat_sampler(train_dataset, int(np.floor(self.valid_size * len(train_dataset))), self.shuffle)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                                   shuffle=False, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory, drop_last=self.drop_last)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                                   shuffle=False, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory, drop_last=self.drop_last)

        if self.attack_sample_size is None:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory)
            calibration_loader = None
        else:
            test_sampler, calibration_sampler = self._creat_sampler(test_dataset, self.attack_sample_size)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory, sampler=test_sampler)
            calibration_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.attack_sample_size,
                                                             shuffle=False, num_workers=self.num_workers,
                                                             pin_memory=self.pin_memory, sampler=calibration_sampler)

        return train_loader, test_loader, valid_loader, calibration_loader



