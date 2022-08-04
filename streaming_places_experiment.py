import argparse
import time
import random

from dataset_utils import make_incremental_features_dataloader, make_features_dataloader, remap_classes
from utils import *
from models.StreamingSoftmaxReplay import StreamingSoftmax
from models.StreamingLDA import StreamingLDA
from models.NCM import NearestClassMean
from models.NaiveBayes import NaiveBayes
from models.OneVsRest import OneVsRest
from models.Perceptron import Perceptron
from models.CBCL import CBCL


def get_class_data_loader(args, class_remap, training, min_class, max_class, batch_size=128, shuffle=False,
                          dataset='places', return_item_ix=False):
    if dataset == 'places' or dataset == 'imagenet' or dataset == 'places_lt':
        h5_file_path = os.path.join(args.h5_features_dir, '%s_features.h5')
        if training:
            data = 'train'
            return_item_ix = return_item_ix
        else:
            data = 'val'
            return_item_ix = False
        return make_incremental_features_dataloader(class_remap, h5_file_path % data, min_class, max_class,
                                                    batch_size=batch_size, shuffle=shuffle,
                                                    return_item_ix=return_item_ix, num_workers=args.num_workers,
                                                    in_memory=args.dataset_in_memory)
    else:
        raise NotImplementedError('Please implement another dataset.')


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_iid_data_loader(args, training, batch_size=128, shuffle=False, dataset='places', return_item_ix=False):
    if dataset == 'places' or dataset == 'imagenet' or dataset == 'places_lt':
        h5_file_path = os.path.join(args.h5_features_dir, '%s_features.h5')
        if training:
            data = 'train'
            return_item_ix = return_item_ix
        else:
            data = 'val'
            return_item_ix = False

        set_seed(args.permutation_seed)  # to return same shuffle
        return make_features_dataloader(h5_file_path % data, batch_size, num_workers=args.num_workers, shuffle=shuffle,
                                        return_item_ix=return_item_ix, in_memory=args.dataset_in_memory)
    else:
        raise NotImplementedError('Please implement another dataset.')


def compute_accuracies(loader, classifier):
    probas, y_test_init = classifier.evaluate_(loader)
    top1, top5 = accuracy(probas, y_test_init, topk=(1, 5))
    return probas, top1, top5


def update_accuracies(class_remap, curr_max_class, classifier, accuracies, save_dir, batch_size, shuffle, dataset):
    seen_classes_test_loader = get_class_data_loader(args, class_remap, False, 0, curr_max_class, batch_size=batch_size,
                                                     shuffle=shuffle, dataset=dataset, return_item_ix=True)
    seen_probas, seen_top1, seen_top5 = compute_accuracies(seen_classes_test_loader, classifier)

    print('\nSeen Classes (%d-%d): top1=%0.2f%% -- top5=%0.2f%%' % (0, curr_max_class - 1, seen_top1, seen_top5))
    accuracies['seen_classes_top1'].append(float(seen_top1))
    accuracies['seen_classes_top5'].append(float(seen_top5))

    # save accuracies and predictions out
    save_accuracies(accuracies, min_class_trained=0, max_class_trained=curr_max_class, save_path=save_dir)
    save_predictions(seen_probas, 0, curr_max_class, save_dir)


def streaming_class_iid_training(args, classifier, class_remap):
    start_time = time.time()
    # start list of accuracies
    accuracies = {'seen_classes_top1': [], 'seen_classes_top5': []}
    # save_name = "model_weights_min_trained_0_max_trained_%d"

    # loop over all data and compute accuracy after every "batch"
    for curr_class_ix in range(0, args.num_classes, args.class_increment):
        max_class = min(curr_class_ix + args.class_increment, args.num_classes)

        # get training loader for current batch
        train_loader = get_class_data_loader(args, class_remap, True, curr_class_ix, max_class,
                                             batch_size=args.batch_size,
                                             shuffle=False, dataset=args.dataset, return_item_ix=True)

        # fit model
        classifier.train_(train_loader)

        if curr_class_ix != 0 and ((curr_class_ix + 1) % args.evaluate_increment == 0):
            # print("\nEvaluating classes from {} to {}".format(0, max_class))
            # output accuracies to console and save out to json file
            update_accuracies(class_remap, max_class, classifier, accuracies, args.save_dir, args.batch_size,
                              shuffle=False, dataset=args.dataset)
            # classifier.save_model(save_dir, save_name % max_class)

    # print final accuracies and time
    test_loader = get_class_data_loader(args, class_remap, False, 0, args.num_classes, batch_size=args.batch_size,
                                        shuffle=False, dataset=args.dataset, return_item_ix=True)
    probas, y_test = classifier.evaluate_(test_loader)
    top1, top5 = accuracy(probas, y_test, topk=(1, 5))
    accuracies['seen_classes_top1'].append(float(top1))
    accuracies['seen_classes_top5'].append(float(top5))

    # save accuracies, predictions, and model out
    save_accuracies(accuracies, min_class_trained=0, max_class_trained=args.num_classes, save_path=args.save_dir)
    save_predictions(probas, 0, args.num_classes, args.save_dir)
    classifier.save_model(args.save_dir, "model_weights_final")

    end_time = time.time()
    print('\nModel Updates: ', classifier.num_updates)
    print('\nFinal: top1=%0.2f%% -- top5=%0.2f%%' % (top1, top5))
    print('\nTotal Time (seconds): %0.2f' % (end_time - start_time))


def streaming_iid_training(args, classifier):
    start_time = time.time()
    # start list of accuracies
    accuracies = {'top1': [], 'top5': []}
    # save_name = "model_weights_%d"

    train_loader = get_iid_data_loader(args, True, batch_size=args.batch_size, shuffle=True, dataset=args.dataset,
                                       return_item_ix=True)
    test_loader = get_iid_data_loader(args, False, batch_size=args.batch_size, shuffle=False, dataset=args.dataset,
                                      return_item_ix=True)

    start = 0
    for batch_x, batch_y, batch_ix in train_loader:
        # fit model
        classifier.fit_batch(batch_x, batch_y, batch_ix)

        end = start + len(batch_x)
        # TODO: decide if we want to compute performance between batches
        start = end

    # print final accuracies and time
    probas, y_test = classifier.evaluate_(test_loader)
    top1, top5 = accuracy(probas, y_test, topk=(1, 5))
    accuracies['top1'].append(float(top1))
    accuracies['top5'].append(float(top5))

    # save accuracies, predictions, and model out
    save_accuracies(accuracies, min_class_trained=0, max_class_trained=args.num_classes, save_path=args.save_dir)
    save_predictions(probas, 0, args.num_classes, args.save_dir)
    classifier.save_model(args.save_dir, "model_weights_final")

    end_time = time.time()
    print('\nModel Updates: ', classifier.num_updates)
    print('\nFinal: top1=%0.2f%% -- top5=%0.2f%%' % (top1, top5))
    print('\nTotal Time (seconds): %0.2f' % (end_time - start_time))


def evaluate(args, classifier):
    start_time = time.time()
    # start list of accuracies
    accuracies = {'top1': [], 'top5': []}
    # save_name = "model_weights_%d"

    test_loader = get_iid_data_loader(args, False, batch_size=args.batch_size, shuffle=False, dataset=args.dataset,
                                      return_item_ix=True)

    # print final accuracies and time
    probas, y_test = classifier.evaluate_(test_loader)
    top1, top5 = accuracy(probas, y_test, topk=(1, 5))
    accuracies['top1'].append(float(top1))
    accuracies['top5'].append(float(top5))

    # save accuracies, predictions, and model out
    # save_accuracies(accuracies, min_class_trained=0, max_class_trained=args.num_classes, save_path=args.save_dir)
    # save_predictions(probas, 0, args.num_classes, args.save_dir)
    # classifier.save_model(args.save_dir, "model_weights_final")

    end_time = time.time()
    print('\nModel Updates: ', classifier.num_updates)
    print('\nFinal: top1=%0.2f%% -- top5=%0.2f%%' % (top1, top5))
    print('\nTotal Time (seconds): %0.2f' % (end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directory parameters
    parser.add_argument('--dataset', type=str, default='places', choices=['places', 'imagenet', 'places_lt'])
    parser.add_argument('--h5_features_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--expt_name', type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--evaluate', type=bool_flag, default=False)

    # general parameters
    parser.add_argument('--dataset_in_memory', type=bool_flag, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--arch', type=str,
                        choices=['resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0',
                                 'efficientnet_b1'])
    parser.add_argument('--num_classes', type=int, default=365)  # total number of classes in the dataset
    parser.add_argument('--batch_size', type=int, default=512)  # batch size for testing
    parser.add_argument('--class_increment', type=int, default=1)
    parser.add_argument('--evaluate_increment', type=int, default=75)  # how many classes before evaluation
    parser.add_argument('--cl_model', type=str, default='slda',
                        choices=['slda', 'fine_tune', 'replay', 'ncm', 'nb', 'ovr', 'perceptron', 'cbcl'])
    parser.add_argument('--permutation_seed', type=int, default=0)
    parser.add_argument('--data_ordering', default='class_iid', choices=['class_iid', 'iid'])

    # SLDA/Naive Bayes parameters
    parser.add_argument('--streaming_update_sigma', type=bool_flag, default=True)  # true to update covariance online
    parser.add_argument('--shrinkage', type=float, default=1e-4)  # shrinkage for SLDA/Naive Bayes

    # Fine-Tune & Replay parameters
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--buffer_size', type=int, default=7300)
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

    makedirs(args.save_dir)

    args.input_feature_size = get_feature_size(args.arch)

    # setup continual model
    print('\nUsing the %s continual learning model.' % args.cl_model)
    if args.cl_model == 'slda':
        classifier = StreamingLDA(args.input_feature_size, args.num_classes,
                                  shrinkage_param=args.shrinkage, streaming_update_sigma=args.streaming_update_sigma)
    elif args.cl_model == 'fine_tune':
        classifier = StreamingSoftmax(args.input_feature_size, args.num_classes, use_replay=False,
                                      lr=args.lr, weight_decay=args.wd)
        if args.ckpt is not None:
            classifier.load_model(args.ckpt)
    elif args.cl_model == 'replay':
        classifier = StreamingSoftmax(args.input_feature_size, args.num_classes, use_replay=True,
                                      lr=args.lr, weight_decay=args.wd, replay_samples=args.replay_size,
                                      max_buffer_size=args.buffer_size)
        if args.ckpt is not None:
            classifier.load_model(args.ckpt)
    elif args.cl_model == 'nb':
        classifier = NaiveBayes(args.input_feature_size, args.num_classes, shrinkage_param=args.shrinkage)
        if args.ckpt is not None:
            classifier.load_model(args.ckpt)
    elif args.cl_model == 'ncm':
        classifier = NearestClassMean(args.input_feature_size, args.num_classes)
        if args.ckpt is not None:
            classifier.load_model(args.ckpt)
    elif args.cl_model == 'ovr':
        classifier = OneVsRest(args.input_feature_size, args.num_classes)
        if args.ckpt is not None:
            classifier.load_model(args.ckpt)
    elif args.cl_model == 'perceptron':
        classifier = Perceptron(args.input_feature_size, args.num_classes)
        if args.ckpt is not None:
            classifier.load_model(args.ckpt)
    elif args.cl_model == 'cbcl':
        classifier = CBCL(args.input_feature_size, args.num_classes, buffer_size=args.buffer_size,
                          distance_threshold=args.distance_threshold,
                          cluster_removal_approach=args.cluster_removal_approach, topk=args.topk_clusters,
                          weighted_pred=args.weighted_predictions)
        if args.ckpt is not None:
            classifier.load_model(args.ckpt)
    else:
        raise NotImplementedError

    if args.evaluate:
        evaluate(args, classifier)
    else:
        if args.data_ordering == 'class_iid':
            # get class ordering
            class_remap = remap_classes(args.num_classes, args.permutation_seed)
            streaming_class_iid_training(args, classifier, class_remap)
        elif args.data_ordering == 'iid':
            streaming_iid_training(args, classifier)
        else:
            raise NotImplementedError
