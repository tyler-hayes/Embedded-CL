import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from OpenLorisDataset import OpenLorisDataset
from FSIOLDataset import FSIOL


def filter_by_class(labels, seen_classes):
    ixs = []
    for c in seen_classes:
        i = list(np.where(labels == c)[0])
        ixs += i
    return ixs


def get_class_change_boundaries(labels):
    class_change_list = []
    prev_class = labels[0]
    for i, curr_class in enumerate(labels):
        if curr_class != prev_class:
            class_change_list.append(i)
            prev_class = curr_class
    return np.array(class_change_list)


def get_stream_data_loader(images_dir, training, ordering=None, batch_size=128, shuffle=False, augment=False,
                           num_workers=8, seen_classes=None, seed=10, ix=None, label_level='class',
                           dataset='openloris'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if training and augment:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])

    if dataset == 'openloris':
        dataset = OpenLorisDataset(images_dir, train=training, ordering=ordering, transform=transform, seed=seed,
                                   label_level=label_level)
    elif dataset == 'fsiol':
        dataset = FSIOL(images_dir, train=training, ordering=ordering, transform=transform, seed=seed)
    labels = np.array([t for t in dataset.targets])

    if seen_classes is not None:
        indices = filter_by_class(labels, seen_classes)
        sub = Subset(dataset, indices)
        loader = DataLoader(sub, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True,
                            shuffle=shuffle)
    elif ix is not None:
        sub = Subset(dataset, ix)
        loader = DataLoader(sub, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True,
                            shuffle=shuffle)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True,
                            shuffle=shuffle)
    return loader


def select_indices(targets, classes, num_samples, seed=200):
    indices = []
    T = np.array(targets)
    samples_per_class = np.ones(len(classes), dtype=int) * int(num_samples / len(classes))
    samples_per_class[:num_samples - sum(samples_per_class)] += 1
    for n, c in enumerate(classes):
        ind = list(np.where(T == c)[0])
        random.seed(seed)
        random.shuffle(ind)
        indices.extend(ind[:samples_per_class[n]])
    if (len(indices) < num_samples) and (len(ind) > samples_per_class[-1]):
        extra_samples = num_samples - len(indices)
        indices.extend(ind[samples_per_class[-1]:samples_per_class[1] + extra_samples])
    return indices


def select_all_indices(targets, classes):
    indices = []
    T = np.array(targets)
    for n, c in enumerate(classes):
        ind = list(np.where(T == c)[0])
        indices.extend(ind)
    return indices


def get_ood_loaders(test_loader, num_classes, batch_size, ood_num_samples, seen_classes=None, dataset=None,
                    only_explicit_out_data=False):
    test_set = test_loader.dataset
    if ood_num_samples == -1:
        print('Using all test data for OOD')
    else:
        print('Num OOD samples ', ood_num_samples)

    if seen_classes is None:
        in_classes = np.arange(num_classes)

        if dataset == 'stream51':
            # since stream-51 has OOD samples not in its included classes
            out_classes = np.array([num_classes])
        else:
            out_classes = np.array(())
    else:
        print('Making in-loader from these classes: ', seen_classes)
        in_classes = np.array(seen_classes)

        if only_explicit_out_data:
            out_classes = np.array([num_classes])
        else:
            b = list(np.arange(num_classes))
            out_classes = np.array([item for item in b if item not in seen_classes])
            if dataset == 'stream51':
                # since stream-51 has OOD samples not in its included classes
                out_classes = np.append(out_classes, num_classes)

    if ood_num_samples == -1:
        in_indices = select_all_indices(test_set.targets, in_classes)
    else:
        in_indices = select_indices(test_set.targets, in_classes, ood_num_samples)
    in_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_set, in_indices), batch_size=batch_size,
                                            shuffle=False, num_workers=8, pin_memory=True)
    res_in_classes = list(np.unique(np.array(test_set.targets)[in_indices]))
    print('Inloader made from {} classes ({} samples)'.format(len(res_in_classes), len(in_indices)))
    print('Inloader made from {} classes ({} samples)'.format(len(res_in_classes), len(in_loader.dataset)))

    if len(out_classes) != 0:

        if ood_num_samples == -1:
            out_indices = select_all_indices(test_set.targets, out_classes)
        else:
            out_indices = select_indices(test_set.targets, out_classes, ood_num_samples)
        out_loader_intra = torch.utils.data.DataLoader(torch.utils.data.Subset(test_set, out_indices),
                                                       batch_size=batch_size,
                                                       shuffle=False, num_workers=8, pin_memory=True)
        res_out_classes = list(np.unique(np.array(test_set.targets)[out_indices]))
        print('Outloader made from {} classes ({} samples)'.format(len(res_out_classes), len(out_indices)))

    else:
        out_loader_intra = None

    return in_loader, out_loader_intra


def auosc_score(in_scores, out_scores, in_correct_label, steps=10000):
    """
    Compute the AUOSC metric that accounts for correct classification and OOD detection.
    :param in_scores: scores for in-distribution samples
    :param out_scores: scores for out-of-distribution samples
    :param in_correct_label: binary vector that says whether the model was correct for each in-distribution sample
    :param steps: number of steps to use for trapezoidal rule calculation
    :return: AUOSC, normalized AUOSC, correct classification rate, false positive rate
    """
    tmin = torch.min(torch.cat((in_scores, out_scores)))
    tmax = torch.max(torch.cat((in_scores, out_scores)))

    if tmin == tmax:
        print('\nWarning: all output scores for in and out samples are identical.')
    thresh_range = torch.arange(tmin, tmax, (tmax - tmin) / steps)
    FPR = []
    CCR = []
    for t in thresh_range:
        FPR.append(float(torch.sum(out_scores > t)) / out_scores.shape[0])
        CCR.append(float(torch.sum(in_correct_label[in_scores > t])) / in_scores.shape[0])
    ccr = np.array(CCR[::-1])
    fpr = np.array(FPR[::-1])
    return np.trapz(ccr, fpr), np.trapz(ccr / ccr.max(), fpr), ccr, fpr


def accuracy(output, target, topk=(1,), output_has_class_ids=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    output = output.cpu()
    target = target.cpu()
    if not output_has_class_ids:
        output = torch.Tensor(output)
    else:
        output = torch.LongTensor(output)
    target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.shape[0]
        if not output_has_class_ids:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = output[:, :maxk].t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def ood_detection(ood_scores_in, true_labels_in, ood_scores_out, true_labels_out, save_dir, index, ood_type):
    # compute auosc scores
    s_in, p_in = ood_scores_in.max(dim=1)
    s_out, _ = ood_scores_out.max(dim=1)
    in_correct_label = p_in == true_labels_in
    auosc, auosc_norm, ccr, fpr = auosc_score(s_in, s_out, in_correct_label)

    num_in_samples = len(ood_scores_in.numpy())
    num_out_samples = len(ood_scores_out.numpy())

    # concatenate everything together
    y_true_ood = np.concatenate((np.ones(num_in_samples), np.zeros(num_out_samples)))
    y_true_actual_labels = np.concatenate((true_labels_in.numpy(), true_labels_out.numpy()))
    y_scores = np.concatenate((ood_scores_in.numpy(), ood_scores_out.numpy()))

    # save out data to calculate metrics offline
    save_array(y_true_actual_labels, index=index, save_path=save_dir,
               save_name='ood_y_true_labels_' + ood_type)
    save_array(y_true_ood, index=index, save_path=save_dir, save_name='ood_y_true_binary_' + ood_type)
    save_array(y_scores, index=index, save_path=save_dir, save_name='ood_y_scores_' + ood_type)

    # compute auroc score and display all ood scores to user
    top_scores = np.max(y_scores, axis=1)
    auroc = roc_auc_score(y_true_ood, top_scores)
    print('\nOOD: AUROC=%0.2f -- AUOSC=%0.2f -- AUOSC Norm=%0.2f' % (
        auroc, auosc, auosc_norm))
    return auroc, auosc, auosc_norm


def update_and_save_accuracies_ood(probas, y_test_init, ood_scores_in_intra, true_labels_in_intra, ood_scores_out_intra,
                                   true_labels_out_intra, save_dir, index, accuracies):
    top1, top5 = accuracy(probas, y_test_init, topk=(1, 5))
    print('\nIndex: %d -- top1=%0.2f%% -- top5=%0.2f%%' % (index, top1, top5))
    accuracies['top1'].append(float(top1))
    accuracies['top5'].append(float(top5))

    auroc_intra, auosc_intra, auosc_norm_intra = ood_detection(ood_scores_in_intra, true_labels_in_intra,
                                                               ood_scores_out_intra, true_labels_out_intra, save_dir,
                                                               index, 'intra')

    accuracies['auroc_score_intra'].append(float(auroc_intra))
    accuracies['auosc_score_intra'].append(float(auosc_intra))
    accuracies['auosc_norm_score_intra'].append(float(auosc_norm_intra))

    # save out results
    save_accuracies(accuracies, index=index, save_path=save_dir)
    save_predictions(probas, index=index, save_path=save_dir)


def update_and_save_accuracies(probas, y_test_init, save_dir, index, accuracies):
    top1, top5 = accuracy(probas, y_test_init, topk=(1, 5))
    print('\nIndex: %d -- top1=%0.2f%% -- top5=%0.2f%%' % (index, top1, top5))
    accuracies['top1'].append(float(top1))
    accuracies['top5'].append(float(top5))

    # save out results
    save_accuracies(accuracies, index=index, save_path=save_dir)
    # save_predictions(probas, index=index, save_path=save_dir)


def save_accuracies(accuracies, index, save_path):
    name = 'accuracies_index_' + str(index) + '.json'
    json.dump(accuracies, open(os.path.join(save_path, name), 'w'))


def save_predictions(y_pred, index, save_path):
    name = 'preds_index_' + str(index)
    torch.save(y_pred, save_path + '/' + name + '.pth')


def save_array(y_pred, index, save_path, save_name):
    name = save_name + '_index_' + str(index)
    torch.save(y_pred, save_path + '/' + name + '.pth')
