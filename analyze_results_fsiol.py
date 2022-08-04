import os
import numpy as np
import json
from collections import defaultdict


def average_fsiol_results(results_dir, expt_folder, file, seeds, cl_models, orders, architectures):
    results_dict_5 = defaultdict(dict)
    results_dict_10 = defaultdict(dict)
    for seed in seeds:
        for cl_model in cl_models:
            for order in orders:
                key = 'top1'
                for arch in architectures:
                    results_file = os.path.join(results_dir,
                                                os.path.join(expt_folder % (cl_model, arch, order, seed), file))
                    with open(results_file, 'r') as f:
                        d = json.load(f)
                        acc = d[key][-1]  # last entry is final

                        if order == '5_shot_class_iid':
                            if cl_model not in results_dict_5[arch]:
                                results_dict_5[arch][cl_model] = []
                            results_dict_5[arch][cl_model].append(acc)
                        elif order == '10_shot_class_iid':
                            if cl_model not in results_dict_10[arch]:
                                results_dict_10[arch][cl_model] = []
                            results_dict_10[arch][cl_model].append(acc)

    print('\n5-shot results')
    for k, v in results_dict_5.items():
        print('\narch: ', k)
        for k2, v2 in v.items():
            print(k2 + ': %0.1f' % np.mean(np.array(v2)))

    print('\n10-shot results')
    for k, v in results_dict_10.items():
        print('\narch: ', k)
        for k2, v2 in v.items():
            print(k2 + ': %0.1f' % np.mean(np.array(v2)))


def main(results_dir):
    #################################################################################################################
    # FSIOL Results
    print('\n\nFSIOL:')

    expt_folder = 'streaming_%s_LR_0.001_%s_fsiol_avg_%s_seed_%d'
    file = 'accuracies_index_-1.json'

    seeds = [10, 20, 30]
    cl_models = ['perceptron', 'fine_tune', 'nb', 'ovr', 'cbcl', 'ncm', 'replay', 'slda']
    orders = ['5_shot_class_iid', '10_shot_class_iid']
    architectures = ['mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b1', 'resnet18']

    # analyze results
    average_fsiol_results(results_dir, expt_folder, file, seeds, cl_models, orders, architectures)


if __name__ == '__main__':
    # directory where all experimental results are saved
    results_dir = '/media/tyler/Data/codes/Embedded-CL/results/'

    main(results_dir)
