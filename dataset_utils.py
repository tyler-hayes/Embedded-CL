import os
from PIL import Image
import h5py
import numpy as np

import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_torchvision_full_image_dataset(traindir, valdir, batch_size=256, test_batch_size=256, shuffle=False,
                                        num_workers=8):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(traindir, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transform),
        batch_size=test_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


class LT_Dataset(Dataset):

    def __init__(self, root, txt, transform=None, return_item_ix=False):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.return_item_ix = return_item_ix
        with open(txt) as f:
            for line in f:
                self.samples.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        path = self.samples[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_item_ix:
            return sample, label, index
        else:
            return sample, label


def load_places_lt_full_image_dataset(image_dir, txt_file_root, batch_size=256, test_batch_size=256, num_workers=8,
                                      return_item_ix=False, shuffle=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = LT_Dataset(image_dir, txt_file_root % 'train', transform=transform, return_item_ix=return_item_ix)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                               pin_memory=True, shuffle=shuffle)

    val_dataset = LT_Dataset(image_dir, txt_file_root % 'test', transform=transform, return_item_ix=return_item_ix)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, num_workers=num_workers,
                                             pin_memory=True, shuffle=False)

    return train_loader, val_loader


class FeaturesDatasetInMemory(Dataset):
    def __init__(self, h5_file_path, return_item_ix=False):
        super(FeaturesDatasetInMemory, self).__init__()
        self.h5_file_path = h5_file_path
        with h5py.File(self.h5_file_path, 'r') as h5:
            self.features = np.array(h5['features'])
            self.labels = np.array(h5['labels'], dtype=np.int64)
            self.dataset_len = len(self.features)

        self.return_item_ix = return_item_ix

    def __getitem__(self, index):

        if self.return_item_ix:
            return self.features[index], self.labels[index], int(index)
        else:
            return self.features[index], self.labels[index]

    def __len__(self):
        return self.dataset_len


class FeaturesDataset(Dataset):
    def __init__(self, h5_file_path, return_item_ix=False, transform=None):
        super(FeaturesDataset, self).__init__()
        self.h5_file_path = h5_file_path
        with h5py.File(self.h5_file_path, 'r') as h5:
            # keys = list(h5.keys())

            self.features_key = 'features'
            features = h5['features']
            self.dataset_len = len(features)

        self.return_item_ix = return_item_ix
        self.transform = transform

    def __getitem__(self, index):
        if not hasattr(self, 'features'):
            self.h5 = h5py.File(self.h5_file_path, 'r')
            self.features = self.h5[self.features_key]
            self.labels = self.h5['labels']

        feat = self.features[index]
        if self.transform is not None:
            feat = self.transform(feat)

        if self.return_item_ix:
            return feat, self.labels[index], int(index)
        else:
            return feat, self.labels[index]

    def __len__(self):
        return self.dataset_len


def make_features_dataloader(h5_file_path, batch_size, num_workers=8, shuffle=False, return_item_ix=False,
                             in_memory=True):
    if in_memory:
        dataset = FeaturesDatasetInMemory(h5_file_path, return_item_ix=return_item_ix)
    else:
        dataset = FeaturesDataset(h5_file_path, return_item_ix=return_item_ix)
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, shuffle=shuffle, batch_size=batch_size)
    return loader


def make_incremental_features_dataloader(class_remap, h5_file_path, min_class, max_class, batch_size, num_workers=8,
                                         shuffle=False, return_item_ix=False, in_memory=True):
    # filter labels between min_class and max_class with class_remap
    h5 = h5py.File(h5_file_path, 'r')
    labels = np.array(h5['labels'], dtype=np.int64)

    class_list = []
    for i in range(min_class, max_class):
        class_list.append(class_remap[i])

    curr_idx = filter_by_class(labels, np.array(class_list))

    # make subset dataset with selected classes
    if in_memory:
        dataset = FeaturesDatasetInMemory(h5_file_path, return_item_ix=False)
    else:
        dataset = FeaturesDataset(h5_file_path, return_item_ix=False)
    loader = setup_subset_dataloader(dataset, curr_idx, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                     return_item_ix=return_item_ix)

    return loader


def filter_by_class(labels, class_list):
    ixs = []
    for c in class_list:
        curr_ix = np.where(labels == c)[0]
        ixs.extend(curr_ix.tolist())
    return ixs


class IndexSampler(torch.utils.data.Sampler):
    """Samples elements sequentially, always in the same order.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class PartialDataset(Dataset):
    def __init__(self, data, indices, return_item_ix):
        self.data = data
        self.indices = indices
        self.return_item_ix = return_item_ix

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y = self.data[index]
        if self.return_item_ix:
            return x, y, index
        else:
            return x, y


def setup_subset_dataloader(dataset, idxs, batch_size=256, shuffle=False, sampler=None, batch_sampler=None,
                            num_workers=8, return_item_ix=False):
    if batch_sampler is None and sampler is None:
        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(idxs)
        else:
            sampler = IndexSampler(idxs)

    dataset = PartialDataset(dataset, idxs, return_item_ix)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=True, batch_sampler=batch_sampler, sampler=sampler)
    return loader


def remap_classes(num_classes, seed):
    # get different class permutations

    np.random.seed(seed)
    ix = np.arange(num_classes)
    np.random.shuffle(ix)
    d = {}
    for i, v in enumerate(ix):
        d[i] = v
    return d


if __name__ == '__main__':
    h5_features_dir = '/media/tyler/Data/codes/edge-cl/features/places/supervised_resnet18_places_avg'
    h5_file_path = os.path.join(h5_features_dir, '%s_features.h5') % 'val'
    class_remap = remap_classes(365, 0)
    loader = make_incremental_features_dataloader(class_remap, h5_file_path, 0, 5, 256, num_workers=8, shuffle=False,
                                                  return_item_ix=False)
    print()
