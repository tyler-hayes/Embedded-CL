import os
import os.path
import random
import numpy as np
from PIL import Image
import torch.utils.data as data

object_list = ['ballpoint', 'book', 'candy', 'cereal', 'comb', 'fork', 'glass', 'hairbrush', 'hairclip', 'keyboard',
               'knife', 'lipstick', 'lotion', 'mouse', 'mug', 'plate', 'shampoo', 'soap', 'spatuala', 'spoon',
               'toothbrush', 'toothpaste']


def parse_data(root, ordering, seed):
    random.seed(seed)
    class_order = list(range(len(object_list)))
    random.shuffle(class_order)

    frames_train = {}
    frames_test = []

    random.seed(seed)
    for i, o in enumerate(object_list):
        l = os.listdir(os.path.join(root, o))  # get file names for object
        # l2 = sorted([int(v.split('.')[0]) for v in l])  # sort based on integer name
        # l3 = [str(v) + '.jpg' for v in l2]  # make names strings again (same object appears twice in a row)
        l = [os.path.join(root, os.path.join(o, v)) for v in l]
        l = np.array(l)

        if '5' in ordering:
            # 5-shot
            train = np.array(random.sample(range(0, len(l)), 5))
            test = np.setdiff1d(np.arange(len(l)), train)
        elif '10' in ordering:
            # 10-shot
            train = np.array(random.sample(range(0, len(l)), 10))
            test = np.setdiff1d(np.arange(len(l)), train)
        else:
            raise NotImplementedError

        train_frames = l[train].tolist()
        test_frames = l[test].tolist()
        frames_train[i] = train_frames
        frames_test.extend([(p, i) for p in test_frames])

    frames_train_list = []
    for c in class_order:
        frames_train_list.extend([(p, c) for p in frames_train[c]])

    if ordering == '5_shot_iid' or ordering == '10_shot_iid':
        random.shuffle(frames_train_list)

    return frames_train_list, frames_test


class FSIOL(data.Dataset):
    """FSIOL Dataset Object
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        ordering (string): desired ordering for training dataset: 'instance',
            'class_instance', 'iid', or 'class_iid' (ignored for test dataset)
            (default: None)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        seed: random seed for shuffling classes or instances (default=10)
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, train=True, ordering=None, transform=None, target_transform=None, seed=10):

        frame_list_train, frame_list_test = parse_data(root, ordering, seed)

        self.root = root
        self.loader = default_loader

        if train:
            self.samples = frame_list_train
            self.targets = [s[-1] for s in frame_list_train]
        else:
            self.samples = frame_list_test
            self.targets = [s[-1] for s in frame_list_test]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        index = int(index)
        fpath, target = self.samples[index][0], self.targets[index]
        # print(fpath)
        sample = self.loader(os.path.join(self.root, fpath))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)
