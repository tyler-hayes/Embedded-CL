import os
import os.path
import random
import numpy as np
from PIL import Image
import torch.utils.data as data

object_list = ['bottle_01', 'bottle_02', 'bottle_03', 'bottle_04',
               'bowl_01', 'bowl_02', 'bowl_03', 'bowl_04', 'bowl_05',
               'corkscrew_01',
               'cottonswab_01', 'cottonswab_02',
               'cup_01', 'cup_02', 'cup_03', 'cup_04', 'cup_05', 'cup_06', 'cup_07', 'cup_08', 'cup_10',
               'cushion_01', 'cushion_02', 'cushion_03',
               'glasses_01', 'glasses_02', 'glasses_03', 'glasses_04',
               'knife_01',
               'ladle_01', 'ladle_02', 'ladle_03', 'ladle_04',
               'mask_01', 'mask_02', 'mask_03', 'mask_04', 'mask_05',
               'paper_cutter_01', 'paper_cutter_02', 'paper_cutter_03', 'paper_cutter_04',
               'pencil_01', 'pencil_02', 'pencil_03', 'pencil_04', 'pencil_05',
               'plasticbag_01', 'plasticbag_02', 'plasticbag_03',
               'plug_01', 'plug_02', 'plug_03', 'plug_04',
               'pot_01',
               'scissors_01', 'scissors_02', 'scissors_03',
               'stapler_01', 'stapler_02', 'stapler_03',
               'thermometer_01', 'thermometer_02', 'thermometer_03',
               'toy_01', 'toy_02', 'toy_03', 'toy_04', 'toy_05',
               'nail_clippers_01', 'nail_clippers_02', 'nail_clippers_03',
               'bracelet_01', 'bracelet_02', 'bracelet_03',
               'comb_01', 'comb_02', 'comb_03',
               'umbrella_01', 'umbrella_02', 'umbrella_03',
               'socks_01', 'socks_02', 'socks_03',
               'toothpaste_01', 'toothpaste_02', 'toothpaste_03',
               'wallet_01', 'wallet_02', 'wallet_03',
               'headphone_01', 'headphone_02', 'headphone_03',
               'key_01', 'key_02', 'key_03',
               'battery_01', 'battery_02',
               'mouse_01',
               'pencilcase_01', 'pencilcase_02',
               'tape_01',
               'chopsticks_01', 'chopsticks_02', 'chopsticks_03',
               'notebook_01', 'notebook_02', 'notebook_03',
               'spoon_01', 'spoon_02', 'spoon_03',
               'tissue_01', 'tissue_02', 'tissue_03',
               'clamp_01', 'clamp_02',
               'hat_01', 'hat_02',
               'u_disk_01', 'u_disk_02',
               'swimming_glasses_01'
               ]

factors = ['clutter', 'illumination', 'occlusion', 'pixel']


def make_class_labels(object_list):
    d = {}
    count = 0
    for obj in object_list:
        id = '_'.join(obj.split('_')[:-1])
        if id not in d:
            d[id] = count
            count += 1
    return d


def make_data_list(keys, data_list):
    new_data_list = []
    for k in keys:
        data = data_list[k]
        data_ids = data[:-1]
        for frame in data[-1]:
            new_data_list.append(data_ids + [os.path.join(k, frame)])
    return new_data_list


def instance_ordering(data_list, seed, keys=None):
    # organize data by video
    random.seed(seed)
    if keys is None:
        keys = list(data_list.keys())
    random.shuffle(keys)  # randomly shuffle videos
    new_data_list = make_data_list(keys, data_list)
    return new_data_list


def find_videos_for_class(obj_paths, num_classes, sort_index=-2):
    keys = np.array(list(obj_paths.keys()))
    cls_ids = np.array([l[sort_index] for l in list(obj_paths.values())])
    cls = {}
    for c in range(num_classes):
        if c not in cls:
            cls[c] = []
        ix = np.where(cls_ids == c)[0]
        cls[c] += list(keys[ix])
    return cls


def class_ordering(data_list, class_type, num_classes, seed):
    # organize by class (class_id = data_list[key][-2])
    new_data_list = []
    cls_paths = find_videos_for_class(data_list, num_classes)

    random.seed(seed)
    cls_order = np.arange(num_classes)
    random.shuffle(cls_order)  # shuffle class order
    for class_id in cls_order:
        curr_paths = cls_paths[class_id]
        random.seed(seed)
        random.shuffle(curr_paths)  # shuffle videos in class

        if class_type == 'class_iid':
            # shuffle all class data
            curr_video_data = make_data_list(curr_paths, data_list)
            random.seed(seed)
            random.shuffle(curr_video_data)  # shuffle frames
        else:
            # shuffle clips within class
            curr_video_data = instance_ordering(data_list, seed, keys=curr_paths)
        new_data_list += curr_video_data

    return new_data_list


def check_number_of_instances(new_data_list):
    x = []
    for v in new_data_list:
        p = v[-1].split('/')[-4:-1]
        if p not in x:
            x.append(p)
    return len(x)


def instance_small_ordering(data_list, seed, num_classes):
    # randomly grab one video per class
    new_data_list = []
    cls_paths = find_videos_for_class(data_list, num_classes)

    random.seed(seed)
    cls_order = np.arange(num_classes)
    random.shuffle(cls_order)  # shuffle class order
    for i, class_id in enumerate(cls_order):
        curr_paths = cls_paths[class_id]
        random.seed(seed + i)
        random.shuffle(curr_paths)  # shuffle videos in class
        curr_ix = 0
        curr_video_data = []
        while len(curr_video_data) == 0:  # some instances do not contain frames, so ensure we grab one with frames
            curr_video_data = make_data_list([curr_paths[curr_ix]],
                                             data_list)  # arbitrarily grab instance since instances were shuffled
            curr_ix += 1
        # print('i: %d -- len: %d' % (i, len(curr_video_data)))
        new_data_list += curr_video_data

    assert num_classes == check_number_of_instances(new_data_list)
    return new_data_list


def instance_small_121_ordering(data_list, seed, num_instances):
    # randomly grab one video per class
    new_data_list = []
    instance_paths = find_videos_for_class(data_list, num_instances, sort_index=-3)  # group all instances

    random.seed(seed)
    instance_order = np.arange(num_instances)
    random.shuffle(instance_order)  # shuffle instance order
    for i, instance_id in enumerate(instance_order):
        curr_paths = instance_paths[instance_id]
        random.seed(seed + i)  # add repeatable amount of integer to seed
        random.shuffle(curr_paths)  # shuffle videos for instance
        curr_ix = 0
        curr_video_data = []
        while len(curr_video_data) == 0:  # some instances do not contain frames, so ensure we grab one with frames
            curr_video_data = make_data_list([curr_paths[curr_ix]],
                                             data_list)  # arbitrarily grab instance since instances were shuffled
            curr_ix += 1
        # print('i: %d -- len: %d' % (i, len(curr_video_data)))
        new_data_list += curr_video_data

    assert num_instances == check_number_of_instances(new_data_list)
    return new_data_list


def make_dataset(data_list, data_type, num_classes, ordering='instance', seed=666):
    """
    data_list
    [domain_factor_id, segment_id, object_id, class_id, frame_list]
    """
    if data_type == 'test':
        return make_data_list(list(data_list.keys()), data_list)  # just return frame list
    if ordering not in ['iid', 'class_iid', 'instance', 'class_instance', 'instance_small', 'instance_small_121']:
        raise ValueError('dataset ordering must be one of: "iid", "class_iid", "instance", or "class_instance"')
    if ordering == 'iid':
        # shuffle all data
        frame_list = make_data_list(list(data_list.keys()), data_list)
        random.seed(seed)
        random.shuffle(frame_list)  # shuffle all frames
        return frame_list
    elif ordering == 'instance':
        return instance_ordering(data_list, seed)
    elif ordering == 'instance_small':
        return instance_small_ordering(data_list, seed, num_classes)
    elif ordering == 'instance_small_121':
        return instance_small_121_ordering(data_list, seed, len(object_list))
    elif 'class' in ordering:
        return class_ordering(data_list, ordering, num_classes, seed)


def parse_data(root_path, data_type):
    """
    obj_paths:
    key = path to object instance
    value = [domain_factor_id, segment_id, object_id, class_id, frame_list]
    """
    segments = range(1, 10)
    class_list = make_class_labels(object_list)

    obj_paths = {}
    for i_fac, fac in enumerate(factors):
        for i_seg, seg in enumerate(segments):
            seg = 'segment%d' % seg
            for i_obj, obj in enumerate(object_list):
                obj_id = '_'.join(obj.split('_')[:-1])
                class_id = class_list[obj_id]
                curr_path = os.path.join(root_path, os.path.join(data_type, os.path.join(fac, os.path.join(seg, obj))))
                obj_paths[curr_path] = [i_fac, i_seg, i_obj, class_id, sorted(os.listdir(curr_path))]
    return obj_paths


class OpenLorisDataset(data.Dataset):
    """OpenLORIS Dataset Object
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

    def __init__(self, root, train=True, ordering=None, transform=None, target_transform=None, num_classes=40, seed=10,
                 label_level='class'):

        if train:
            data_type = 'train'
            data_list = parse_data(root, 'train')
        else:
            data_type = 'test'
            data_list = parse_data(root, 'test')

        # print('\nCreating OpenLORIS dataset object for %s with labels at the %s level.' % (data_type, label_level))

        # sort videos by ordering
        frame_list = make_dataset(data_list, data_type, num_classes, ordering=ordering, seed=seed)

        self.root = root
        self.loader = default_loader

        self.samples = frame_list

        if label_level == 'class':
            label_index = -2
        elif label_level == 'instance':
            label_index = -3
        else:
            raise NotImplementedError
        self.targets = [s[label_index] for s in frame_list]

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
        fpath, target = self.samples[index][-1], self.targets[index]
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
