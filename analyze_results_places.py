import os
import numpy as np
import json
from collections import defaultdict


def average_places_results(results_dir, expt_folder, file, seeds, cl_models, orders, architectures, lr):
    lr = str(lr)
    results_dict_iid = defaultdict(dict)
    results_dict_class_iid = defaultdict(dict)
    for seed in seeds:
        for cl_model in cl_models:
            for order in orders:
                if order == 'iid':
                    key = 'top1'
                elif order == 'class_iid':
                    key = 'seen_classes_top1'
                for arch in architectures:
                    results_file = os.path.join(results_dir,
                                                os.path.join(expt_folder % (cl_model, lr, arch, order, seed), file))
                    with open(results_file, 'r') as f:
                        d = json.load(f)
                        acc = d[key][-1]  # last entry is final

                        if order == 'iid':
                            if cl_model not in results_dict_iid[arch]:
                                results_dict_iid[arch][cl_model] = []
                            results_dict_iid[arch][cl_model].append(acc)
                        else:
                            if cl_model not in results_dict_class_iid[arch]:
                                results_dict_class_iid[arch][cl_model] = []
                            results_dict_class_iid[arch][cl_model].append(acc)

    print('\niid')
    for k, v in results_dict_iid.items():
        print('\narch: ', k)
        for k2, v2 in v.items():
            print(k2 + ': %0.1f' % np.mean(np.array(v2)))

    print('\nclass iid')
    for k, v in results_dict_class_iid.items():
        print('\narch: ', k)
        for k2, v2 in v.items():
            print(k2 + ': %0.1f' % np.mean(np.array(v2)))


def main(results_dir):
    #################################################################################################################
    # Places-LT Results
    print('\n\nPlaces-LT:')

    expt_folder = 'streaming_%s_LR_%s_%s_places_lt_avg_%s_seed_%s'
    file = 'accuracies_min_trained_0_max_trained_365.json'

    seeds = [0, 1, 2]
    cl_models = ['fine_tune', 'slda', 'replay', 'replay_2percls', 'ncm', 'ovr', 'nb', 'perceptron']
    orders = ['iid', 'class_iid']
    architectures = ['mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b1', 'resnet18']
    lr = 0.001

    # evaluate results
    average_places_results(results_dir, expt_folder, file, seeds, cl_models, orders, architectures, lr)

    #################################################################################################################
    # Places-365 Results
    print('\n\nPlaces-365:')

    expt_folder = 'streaming_%s_LR_%s_%s_places_avg_%s_seed_%s'
    file = 'accuracies_min_trained_0_max_trained_365.json'

    seeds = [0]
    cl_models = ['fine_tune', 'slda', 'replay', 'replay_2percls', 'ncm', 'nb', 'perceptron']
    orders = ['iid', 'class_iid']
    architectures = ['mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b1', 'resnet18']
    lr = 0.0001

    # evaluate results
    average_places_results(results_dir, expt_folder, file, seeds, cl_models, orders, architectures, lr)


if __name__ == '__main__':
    # directory where all experimental results are saved
    results_dir = '/media/tyler/Data/codes/Embedded-CL/results/'

    main(results_dir)
