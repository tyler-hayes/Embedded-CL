import os
import argparse
import h5py
import numpy as np
import json

import torch

from dataset_utils import load_torchvision_full_image_dataset, load_places_lt_full_image_dataset
from utils import get_backbone, makedirs


def get_data_loaders(args):
    if args.dataset in ['imagenet', 'places']:
        traindir = os.path.join(args.images_dir, 'train')
        valdir = os.path.join(args.images_dir, 'val')
        train_loader, val_loader = load_torchvision_full_image_dataset(traindir, valdir, batch_size=args.batch_size,
                                                                       test_batch_size=args.batch_size,
                                                                       num_workers=args.num_workers, shuffle=False)
    elif args.dataset == 'places_lt':
        train_loader, val_loader = load_places_lt_full_image_dataset(args.images_dir, args.lt_txt_file,
                                                                     batch_size=args.batch_size,
                                                                     test_batch_size=args.batch_size,
                                                                     num_workers=args.num_workers, shuffle=False)
    else:
        raise NotImplementedError
    return train_loader, val_loader


def make_h5_feature_file(dataset, model, loader, h5_file_full_path, data_type, feature_size, device):
    if os.path.exists(h5_file_full_path):
        # os.remove(h5_file_full_path)
        # print('removed old h5 file')
        print('file already exists')
        return
    h5_file = h5py.File(h5_file_full_path, 'w')

    # preset array sizes
    if dataset == 'imagenet':
        num_train = 1281167
        num_val = 50000
    elif dataset == 'places':
        num_train = 1803460
        num_val = 36500
    elif dataset == 'places_lt':
        num_train = 62500
        num_val = 36500
    else:
        raise NotImplementedError

    if data_type == 'train':
        h5_file.create_dataset("features", shape=(num_train, feature_size), dtype=np.float32)
        h5_file.create_dataset("labels", shape=(num_train,), dtype=np.int64)
    elif data_type == 'val':
        h5_file.create_dataset("features", (num_val, feature_size), dtype=np.float32)
        h5_file.create_dataset("labels", (num_val,), dtype=np.int64)
    else:
        raise NotImplementedError

    with torch.no_grad():

        # switch to evaluate mode
        model.eval().to(device)
        start = 0

        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images = images.to(device)
                cur_feats = model(images).cpu()
                cur_targets = target.cpu()
                B, D = cur_feats.shape

                h5_file['features'][start:start + B, :] = cur_feats.numpy()
                h5_file['labels'][start:start + B] = cur_targets.numpy()

                start += B
    h5_file.close()


def cache_features(args):
    train_loader, val_loader = get_data_loaders(args)
    backbone, feature_size = get_backbone(args.arch, args.pooling_type)

    print('\ncaching val features...')
    make_h5_feature_file(args.dataset, backbone, val_loader, os.path.join(args.cache_h5_dir, 'val_features.h5'), 'val',
                         feature_size, args.device)
    print('\ncaching train features...')
    make_h5_feature_file(args.dataset, backbone, train_loader, os.path.join(args.cache_h5_dir, 'train_features.h5'),
                         'train', feature_size, args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directory parameters
    parser.add_argument('--dataset', type=str, default='places', choices=['places', 'imagenet', 'places_lt'])
    parser.add_argument('--images_dir', type=str)  # path to images (folder with 'train' and 'val' for data)
    parser.add_argument('--cache_h5_dir', type=str, default=None)
    parser.add_argument('--lt_txt_file', type=str,
                        default='/media/tyler/Data/datasets/Places-LT/Places_LT_%s.txt')

    # other parameters
    parser.add_argument('--arch', type=str,
                        choices=['resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0',
                                 'efficientnet_b1'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--pooling_type', type=str, default='avg', choices=['avg', 'max'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # if not os.path.exists(args.cache_h5_dir):
    #     os.mkdir(args.cache_h5_dir)
    makedirs(args.cache_h5_dir)

    cache_features(args)
