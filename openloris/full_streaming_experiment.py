import os
import argparse
import json
import time
import numpy as np
import torch

from open_loris_utils import get_stream_data_loader, get_class_change_boundaries, get_ood_loaders, \
    update_and_save_accuracies_ood, save_accuracies, update_and_save_accuracies
from models.StreamingLDA import StreamingLDA
from models.StreamingSoftmaxReplay import StreamingSoftmax
from models.OneVsRest import OneVsRest
from models.NaiveBayes import NaiveBayes
from models.NCM import NearestClassMean
from models.Perceptron import Perceptron
from models.CBCL import CBCL
from utils import get_backbone, bool_flag


def run_experiment(dataset, images_dir, order, num_classes, classifier, save_dir, step_size, batch_size=256,
                   augment=False, ood_num_samples=1000, seed=200, device='cuda', perform_ood=False,
                   label_level='class'):
    start_time = time.time()
    if dataset == 'openloris':
        test_order = None
    elif dataset == 'fsiol':
        test_order = order

    if perform_ood:
        accuracies = {'top1': [], 'top5': [], 'auroc_score_intra': [], 'auosc_score_intra': [],
                      'auosc_norm_score_intra': [], 'time': []}
    else:
        accuracies = {'top1': [], 'top5': [], 'time': []}

    if order == 'instance_small' or order == 'instance_small_121':
        # get instance level boundaries
        tr = get_stream_data_loader(images_dir, True, ordering=order, batch_size=batch_size, shuffle=False,
                                    augment=augment, seed=seed, label_level='instance', dataset=dataset)
    else:
        tr = get_stream_data_loader(images_dir, True, ordering=order, batch_size=batch_size, shuffle=False,
                                    augment=augment, seed=seed, label_level=label_level, dataset=dataset)
    test_loader_full = get_stream_data_loader(images_dir, False, ordering=test_order, batch_size=batch_size,
                                              seen_classes=None, label_level=label_level, dataset=dataset)

    if order in ['iid', 'instance'] and label_level == 'class':
        eval_ix = np.arange(step_size, len(tr.dataset), step_size)
    # elif order in ['class_iid', 'class_instance'] or (label_level == 'instance' and order == 'instance'):
    elif order in ['5_shot_iid', '5_shot_class_iid', '10_shot_iid', '10_shot_class_iid']:
        eval_ix = np.array([])  # just evaluate at the end
    else:
        train_labels = np.array([t for t in tr.dataset.targets])
        eval_ix = get_class_change_boundaries(train_labels)
        if step_size != 1:
            eval_ix = eval_ix[step_size - 1:-1:step_size - 1]

    # add final point for last evaluation
    eval_ix = np.append(eval_ix, np.array(len(tr.dataset)))
    print('eval_ix ', eval_ix)
    print('len eval_ix ', len(eval_ix))

    if order == 'instance_small' or order == 'instance_small_121':
        # get loader with class level labels
        tr = get_stream_data_loader(images_dir, True, ordering=order, batch_size=batch_size, shuffle=False,
                                    augment=augment, seed=seed, label_level=label_level, dataset=dataset)

    print('Beginning streaming training...')
    seen_classes = []
    i = 0
    for batch_ix, (batch_x, batch_y) in enumerate(tr):

        if classifier.backbone is not None:
            batch_x_feat = classifier.backbone(batch_x.to(device)).detach()
        else:
            batch_x_feat = batch_x.to(device).detach()

        for x, y in zip(batch_x_feat, batch_y):

            if i in eval_ix and i != 0:

                if order in ['iid', 'instance'] and label_level == 'class':
                    test_classes = np.arange(num_classes)
                else:
                    test_classes = seen_classes

                print('Making test loader from following: ', test_classes)
                test_loader = get_stream_data_loader(images_dir, False, ordering=test_order, batch_size=batch_size,
                                                     seen_classes=test_classes, label_level=label_level,
                                                     dataset=dataset)
                probas, y_test_init = classifier.evaluate_(test_loader)

                if perform_ood:
                    in_loader, out_loader_intra = get_ood_loaders(test_loader_full, num_classes,
                                                                  batch_size, ood_num_samples,
                                                                  seen_classes=seen_classes,
                                                                  dataset=dataset)
                    ood_scores_in, true_labels_in = classifier.evaluate_ood_(in_loader)
                    ood_scores_out, true_labels_out = classifier.evaluate_ood_(out_loader_intra)

                    update_and_save_accuracies_ood(probas, y_test_init, ood_scores_in, true_labels_in, ood_scores_out,
                                                   true_labels_out, save_dir, i, accuracies)
                else:
                    update_and_save_accuracies(probas, y_test_init, save_dir, i, accuracies)
                    classifier.save_model(save_dir, "model_weights_%d" % i)

            # fit model
            classifier.fit(x, y.view(1, ), torch.tensor(i).long())
            i += 1

            # if class not yet in seen_classes, append it
            if y.item() not in seen_classes:
                seen_classes.append(y.item())

    print('Making test loader from following: ', seen_classes)
    test_loader = get_stream_data_loader(images_dir, False, ordering=test_order, batch_size=batch_size,
                                         seen_classes=seen_classes, label_level=label_level, dataset=dataset)
    probas, y_test_init = classifier.evaluate_(test_loader)

    if perform_ood:
        in_loader, out_loader_intra = get_ood_loaders(test_loader_full, num_classes,
                                                      batch_size, ood_num_samples,
                                                      seen_classes=seen_classes,
                                                      dataset=dataset)
        ood_scores_in, true_labels_in = classifier.evaluate_ood_(in_loader)
        ood_scores_out, true_labels_out = classifier.evaluate_ood_(out_loader_intra)

        update_and_save_accuracies_ood(probas, y_test_init, ood_scores_in, true_labels_in, ood_scores_out,
                                       true_labels_out,
                                       save_dir, i, accuracies)
    else:
        update_and_save_accuracies(probas, y_test_init, save_dir, i, accuracies)

    end_time = time.time()
    accuracies['time'].append(end_time - start_time)
    save_accuracies(accuracies, index=-1, save_path=save_dir)
    classifier.save_model(save_dir, "model_weights_final")
    return accuracies


def evaluate(dataset, images_dir, order, num_classes, classifier, save_dir, step_size, batch_size=256,
             augment=False, ood_num_samples=1000, seed=200, device='cuda', perform_ood=False,
             label_level='class'):
    start_time = time.time()
    if dataset == 'openloris':
        test_order = None
    elif dataset == 'fsiol':
        test_order = order

    if perform_ood:
        accuracies = {'top1': [], 'top5': [], 'auroc_score_intra': [], 'auosc_score_intra': [],
                      'auosc_norm_score_intra': [], 'time': []}
    else:
        accuracies = {'top1': [], 'top5': [], 'time': []}

    if order == 'instance_small' or order == 'instance_small_121':
        # get instance level boundaries
        tr = get_stream_data_loader(images_dir, True, ordering=order, batch_size=batch_size, shuffle=False,
                                    augment=augment, seed=seed, label_level='instance', dataset=dataset)
    else:
        tr = get_stream_data_loader(images_dir, True, ordering=order, batch_size=batch_size, shuffle=False,
                                    augment=augment, seed=seed, label_level=label_level, dataset=dataset)
    test_loader_full = get_stream_data_loader(images_dir, False, ordering=test_order, batch_size=batch_size,
                                              seen_classes=None, label_level=label_level, dataset=dataset)

    if order in ['iid', 'instance'] and label_level == 'class':
        eval_ix = np.arange(step_size, len(tr.dataset), step_size)
    # elif order in ['class_iid', 'class_instance'] or (label_level == 'instance' and order == 'instance'):
    else:
        train_labels = np.array([t for t in tr.dataset.targets])
        eval_ix = get_class_change_boundaries(train_labels)
        if step_size != 1:
            eval_ix = eval_ix[step_size - 1:-1:step_size - 1]

    # add final point for last evaluation
    eval_ix = np.append(eval_ix, np.array(len(tr.dataset)))
    print('eval_ix ', eval_ix)
    print('len eval_ix ', len(eval_ix))

    if order == 'instance_small' or order == 'instance_small_121':
        # get loader with class level labels
        tr = get_stream_data_loader(images_dir, True, ordering=order, batch_size=batch_size, shuffle=False,
                                    augment=augment, seed=seed, label_level=label_level, dataset=dataset)

    print('Beginning streaming training...')
    seen_classes = []
    i = 0
    for batch_ix, (batch_x, batch_y) in enumerate(tr):

        for y in batch_y:

            if i in eval_ix and i != 0:

                if order in ['iid', 'instance'] and label_level == 'class':
                    test_classes = np.arange(num_classes)
                else:
                    test_classes = seen_classes

                print('Making test loader from following: ', test_classes)
                test_loader = get_stream_data_loader(images_dir, False, ordering=test_order, batch_size=batch_size,
                                                     seen_classes=test_classes, label_level=label_level,
                                                     dataset=dataset)
                classifier.load_model(os.path.join(save_dir, "model_weights_%d" % i + '.pth'))
                probas, y_test_init = classifier.evaluate_(test_loader)

                if perform_ood:
                    in_loader, out_loader_intra = get_ood_loaders(test_loader_full, num_classes,
                                                                  batch_size, ood_num_samples,
                                                                  seen_classes=seen_classes,
                                                                  dataset=dataset)
                    ood_scores_in, true_labels_in = classifier.evaluate_ood_(in_loader)
                    ood_scores_out, true_labels_out = classifier.evaluate_ood_(out_loader_intra)

                    update_and_save_accuracies_ood(probas, y_test_init, ood_scores_in, true_labels_in, ood_scores_out,
                                                   true_labels_out, save_dir, i, accuracies)
                else:
                    update_and_save_accuracies(probas, y_test_init, save_dir, i, accuracies)
                    # classifier.save_model(save_dir, "model_weights_%d" % i)

            i += 1

            # if class not yet in seen_classes, append it
            if y.item() not in seen_classes:
                seen_classes.append(y.item())

    print('Making test loader from following: ', seen_classes)
    test_loader = get_stream_data_loader(images_dir, False, ordering=test_order, batch_size=batch_size,
                                         seen_classes=seen_classes, label_level=label_level, dataset=dataset)
    classifier.load_model(os.path.join(save_dir, "model_weights_final.pth"))
    probas, y_test_init = classifier.evaluate_(test_loader)

    if perform_ood:
        in_loader, out_loader_intra = get_ood_loaders(test_loader_full, num_classes,
                                                      batch_size, ood_num_samples,
                                                      seen_classes=seen_classes,
                                                      dataset=dataset)
        ood_scores_in, true_labels_in = classifier.evaluate_ood_(in_loader)
        ood_scores_out, true_labels_out = classifier.evaluate_ood_(out_loader_intra)

        update_and_save_accuracies_ood(probas, y_test_init, ood_scores_in, true_labels_in, ood_scores_out,
                                       true_labels_out,
                                       save_dir, i, accuracies)
    else:
        update_and_save_accuracies(probas, y_test_init, save_dir, i, accuracies)

    end_time = time.time()
    accuracies['time'].append(end_time - start_time)
    save_accuracies(accuracies, index=-1, save_path=save_dir)
    # classifier.save_model(save_dir, "model_weights_final")
    return accuracies


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='openloris', choices=['openloris', 'fsiol'])
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--expt_name', type=str)
    parser.add_argument('--order', type=str,
                        choices=['iid', 'class_iid', 'instance', 'class_instance', 'instance_small',
                                 'instance_small_121', '5_shot_iid', '5_shot_class_iid', '10_shot_iid',
                                 '10_shot_class_iid'])
    parser.add_argument('--label_level', type=str, choices=['class', 'instance'], default='class')
    parser.add_argument('--evaluate', type=bool_flag, default=False)

    parser.add_argument('--model', type=str, default='slda',
                        choices=['slda', 'fine_tune', 'replay', 'ovr', 'ncm', 'nb', 'perceptron', 'cbcl'])
    parser.add_argument('--arch', type=str,
                        choices=['resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0',
                                 'efficientnet_b1'])
    parser.add_argument('--pooling_type', type=str, choices=['avg', 'max'], default='avg')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--ood_num_samples', type=int, default=-1)
    parser.add_argument('--perform_ood', type=bool_flag, default=False)  # true to update covariance online

    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10)

    # SLDA parameters
    parser.add_argument('--streaming_update_sigma', type=bool_flag, default=True)  # true to update covariance online
    parser.add_argument('--shrinkage', type=float, default=1e-4)  # shrinkage for SLDA
    parser.add_argument('--slda_ood_type', type=str, default='baseline', choices=['mahalanobis', 'baseline'])

    # Fine-Tune & Replay parameters
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--buffer_size', type=int, default=800)
    parser.add_argument('--replay_size', type=int, default=50)

    # CBCL parameters
    parser.add_argument('--cluster_removal_approach', type=str, default='min_dist', choices=['min_dist', 'max'])
    parser.add_argument('--distance_threshold', type=float, default=70)
    parser.add_argument('--topk_clusters', type=int, default=1)
    parser.add_argument('--weighted_predictions', type=bool_flag, default=True)

    args = parser.parse_args()
    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    if args.save_dir is None:
        args.save_dir = 'streaming_experiments/' + args.expt_name
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define backbone
    backbone, feature_size = get_backbone(args.arch, args.pooling_type)

    # setup continual model
    print('\nUsing the %s continual learning model.' % args.model)
    if args.model == 'slda':
        classifier = StreamingLDA(feature_size, args.num_classes, backbone=backbone,
                                  shrinkage_param=args.shrinkage, streaming_update_sigma=args.streaming_update_sigma,
                                  ood_type=args.slda_ood_type)
    elif args.model == 'fine_tune':
        classifier = StreamingSoftmax(feature_size, args.num_classes, use_replay=False, backbone=backbone,
                                      lr=args.lr, weight_decay=args.wd)
    elif args.model == 'replay':
        classifier = StreamingSoftmax(feature_size, args.num_classes, use_replay=True, backbone=backbone,
                                      lr=args.lr, weight_decay=args.wd, replay_samples=args.replay_size,
                                      max_buffer_size=args.buffer_size)

    elif args.model == 'nb':
        classifier = NaiveBayes(feature_size, args.num_classes, shrinkage_param=args.shrinkage, backbone=backbone)
    elif args.model == 'ncm':
        classifier = NearestClassMean(feature_size, args.num_classes, backbone=backbone)
    elif args.model == 'ovr':
        classifier = OneVsRest(feature_size, args.num_classes, backbone=backbone)
    elif args.model == 'perceptron':
        classifier = Perceptron(feature_size, args.num_classes, backbone=backbone)
    elif args.model == 'cbcl':
        classifier = CBCL(feature_size, args.num_classes, buffer_size=args.buffer_size, backbone=backbone,
                          distance_threshold=args.distance_threshold,
                          cluster_removal_approach=args.cluster_removal_approach, topk=args.topk_clusters,
                          weighted_pred=args.weighted_predictions)
    else:
        raise NotImplementedError

    # perform streaming classification
    if args.evaluate:
        evaluate(args.dataset, args.images_dir, args.order, args.num_classes, classifier, args.save_dir, args.step,
                 ood_num_samples=args.ood_num_samples, seed=args.seed, batch_size=args.batch_size,
                 perform_ood=args.perform_ood, label_level=args.label_level)
    else:
        run_experiment(args.dataset, args.images_dir, args.order, args.num_classes, classifier, args.save_dir,
                       args.step,
                       ood_num_samples=args.ood_num_samples, seed=args.seed, batch_size=args.batch_size,
                       perform_ood=args.perform_ood, label_level=args.label_level)


if __name__ == '__main__':
    main()
