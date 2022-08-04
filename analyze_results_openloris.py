import os
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import makedirs


def compute_memory_requirements(arch, model, num_classes, include_bb=False):
    if arch == 'mobilenet_v3_small':
        fs = 576
        bb_params = 927008
    elif arch == 'mobilenet_v3_large':
        fs = 960
        bb_params = 2971952
    elif arch == 'efficientnet_b0':
        fs = 1280
        bb_params = 4007548
    elif arch == 'efficientnet_b1':
        fs = 1280
        bb_params = 6513184
    elif arch == 'resnet18':
        fs = 512
        bb_params = 11176512
    else:
        raise NotImplementedError

    if model in ['fine_tune', 'ovr', 'ncm', 'perceptron']:
        params = fs * num_classes
    elif model in ['slda']:
        params = fs * num_classes + fs * fs
    elif model in ['nb', 'replay_2percls']:
        params = 3 * fs * num_classes
    elif model in ['replay']:
        params = 21 * fs * num_classes
    elif model in ['cbcl']:
        params = 20 * fs * num_classes
    else:
        raise NotImplementedError

    if include_bb:
        return params + bb_params
    else:
        return params


def average_openloris_instance_results(results_dir, expt_folder, file, seeds, cl_models, orders, architectures, lr,
                                       print_results=True):
    for order in orders:
        lr = str(lr)
        results_dict = defaultdict(dict)
        for seed in seeds:
            for cl_model in cl_models:
                key = 'top1'
                for arch in architectures:
                    results_file = os.path.join(results_dir,
                                                os.path.join(expt_folder % (cl_model, lr, arch, order, seed), file))
                    with open(results_file, 'r') as f:
                        d = json.load(f)
                        acc = d[key][-1]  # last entry is final

                        if cl_model not in results_dict[arch]:
                            results_dict[arch][cl_model] = []
                        results_dict[arch][cl_model].append(acc)

        if print_results:
            print('\nOrder: ', order)
            for k, v in results_dict.items():
                print('Arch: ', k)
                for k2, v2 in v.items():
                    print(k2 + ': %0.1f' % np.mean(np.array(v2)))

    return results_dict


def make_small_instance_plot(results_dict, pretty_names, save_dir=None, save_name=None, include_std=True):
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']
    fig, ax = plt.subplots()
    x = np.arange(40)

    for i, (label, values) in enumerate(results_dict.items()):
        label = pretty_names[label]
        v_mu = values[0]
        v_stdev = values[1]
        ax.plot(x, v_mu, marker='o', linewidth=2, markersize=5, label=label, color=CB_color_cycle[i])
        if include_std:
            ax.fill_between(x, v_mu - v_stdev, v_mu + v_stdev, alpha=0.35, color=CB_color_cycle[i])

    plt.xlabel('Number of Instances Trained', fontweight='bold', fontsize=14)
    plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim([0, 105])
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=[0.5, 1.4], ncol=3, fancybox=True, shadow=True)
    plt.grid()
    plt.tight_layout()
    plt.show()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, save_name + '.png'), bbox_inches="tight", format='png', dpi=600)


def compute_metric(acc, params, time, alpha=2, beta=0.25, gamma=0.25, s=20):
    return s * np.log(acc ** alpha / (params ** beta * time ** gamma))


def plot_open_loris(cl_models, pretty_names_bar, pretty_names_learning_curve, architectures, results_dir, expt_folder,
                    file, seeds, orders, lr, save_dir, alpha=2, beta=0.25, gamma=0.25, num_classes=40, include_bb=True):
    results_dict = defaultdict(dict)
    timing_dict = defaultdict(dict)
    for seed in seeds:
        for cl_model in cl_models:
            for order in orders:
                if order == 'iid' or order == 'instance' or order == 'instance_small':
                    key = 'top1'
                elif order == 'class_iid':
                    key = 'seen_classes_top1'
                else:
                    raise NotImplementedError
                for arch in architectures:
                    results_file = os.path.join(results_dir,
                                                os.path.join(expt_folder % (cl_model, lr, arch, order, seed), file))
                    with open(results_file, 'r') as f:
                        d = json.load(f)
                        acc = d[key]
                        time = d['time']
                        if cl_model not in results_dict[arch]:
                            results_dict[arch][cl_model] = []
                            timing_dict[arch][cl_model] = []
                        results_dict[arch][cl_model].append(acc)
                        timing_dict[arch][cl_model].append(time)

    for arch, results in results_dict.items():
        res = {}
        for k, v in results.items():
            res[k] = (np.mean(np.array(v), axis=0), np.std(np.array(v), axis=0) / np.sqrt(len(v)))
        make_small_instance_plot(res, pretty_names_learning_curve, save_dir,
                                 save_name='low_shot_instance_open_loris_%s' % arch)

    axes_dict = defaultdict(dict)
    low_shot_dict = defaultdict(dict)
    for arch in architectures:
        for k, v in results_dict[arch].items():
            params = compute_memory_requirements(arch, k, num_classes=num_classes, include_bb=include_bb)
            if k not in axes_dict[arch]:
                axes_dict[arch][k] = []
                low_shot_dict[arch][k] = []
            low_shot_dict[arch][k].append(np.mean(np.array([l[-1] for l in v])))
            axes_dict[arch][k].append(np.mean(np.array([l[-1] for l in v])))
            axes_dict[arch][k].append(params)
        for k, v in timing_dict[arch].items():
            axes_dict[arch][k].append(np.mean(np.array([l[-1] for l in v])))

    print('\nEFFICACY & MEMORY & COMPUTE & Ω')
    for arch, dict_ in axes_dict.items():
        print('\nArchitecture:', arch)
        for model, vals in dict_.items():
            m = compute_metric(vals[0], vals[1], vals[2], alpha=alpha, beta=beta, gamma=gamma)
            print('%s & %0.1f & %d & %d & %0.1f' % (
                model, vals[0], vals[1], vals[2], m))

    r = average_openloris_instance_results(results_dir, expt_folder, file, seeds, cl_models, ['instance'],
                                           architectures, lr, print_results=False)
    d3 = defaultdict(dict)
    for a in architectures:
        for m in cl_models:
            if m not in d3[a]:
                d3[a][m] = []
            d3[a][m].append(np.mean(r[a][m]))

    d4 = {}
    for mp in cl_models:
        low_shot_perf = np.array([v[mp] for k, v in low_shot_dict.items()])
        full_perf = np.array([v[mp] for k, v in d3.items()])
        d4[mp] = [np.mean(full_perf), np.mean(low_shot_perf)]

    make_bar_chart(d4, pretty_names_bar, save_dir, 'openloris_bar_chart')


def make_bar_chart(d, pretty_names, save_dir=None, save_name=None):
    colors = ['#D81B60', '#648FFF']

    labels_orig = list(d.keys())
    labels = []
    for l in labels_orig:
        labels.append(pretty_names[l])

    full_instance = [f[0] for f in d.values()]
    low_shot_instance = [f[1] for f in d.values()]

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = plt.bar(x - width / 2, full_instance, width, label='Instance', color=colors[0], alpha=1.0)
    rects2 = plt.bar(x + width / 2, low_shot_instance, width, label='Low-Shot Instance', color=colors[1], alpha=1.0,
                     hatch='/')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Average Final Accuracy [%]', fontweight='bold', fontsize=16)
    plt.xticks(x, labels, fontsize=14)
    plt.yticks(fontsize=18)
    plt.ylim([0, 104])
    ax.legend(["Instance", "Low-Shot Instance"], loc='upper center',
              bbox_to_anchor=(0.5, 1.15),
              ncol=2, fancybox=True, shadow=True, fontsize=14)

    ax.bar_label(rects1, padding=2, fmt='%.1f', fontsize=13)
    ax.bar_label(rects2, padding=2, fmt='%.1f', fontsize=13)

    fig.tight_layout()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)

    plt.show()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, save_name + '.png'), bbox_inches="tight", format='png', dpi=600)


def main(results_dir, save_dir):
    #################################################################################################################
    # OpenLORIS Results
    print('\n\nOpenLORIS:')

    expt_folder = 'streaming_%s_LR_%s_%s_openloris_avg_%s_seed_%d'
    file = 'accuracies_index_-1.json'

    seeds = [10, 20, 30]
    cl_models = ['fine_tune', 'slda', 'replay', 'replay_2percls', 'ncm', 'ovr', 'nb', 'perceptron']
    orders = ['instance', 'instance_small']
    architectures = ['mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b1', 'resnet18']
    lr = 0.001

    # evaluate results
    average_openloris_instance_results(results_dir, expt_folder, file, seeds, cl_models, orders, architectures, lr)

    #################################################################################################################
    # OpenLORIS Plots
    print('\n\nOpenLORIS (Plots):')

    expt_folder = 'streaming_%s_LR_%s_%s_openloris_avg_%s_seed_%d'
    makedirs(save_dir)
    file = 'accuracies_index_-1.json'

    seeds = [10, 20, 30]
    cl_models = ['perceptron', 'fine_tune', 'nb', 'ovr', 'ncm', 'replay_2percls', 'replay', 'slda']
    orders = ['instance_small']
    architectures = ['mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b1', 'resnet18']
    lr = 0.001

    # pretty names for plots
    pretty_names_bar = ['Perceptron', 'Fine\nTune', 'Naive\nBayes', 'SOvR', 'NCM', 'Replay\n(2pc)',
                        'Replay\n(20pc)', 'SLDA']
    pretty_names_learning_curves = ['Perceptron', 'Fine-Tune', 'Naive Bayes', 'SOvR', 'NCM', 'Replay (2pc)',
                                    'Replay (20pc)', 'SLDA']
    pretty_names_bar_dict = {}
    for orig, new in zip(cl_models, pretty_names_bar):
        pretty_names_bar_dict[orig] = new

    pretty_names_learning_curves_dict = {}
    for orig, new in zip(cl_models, pretty_names_learning_curves):
        pretty_names_learning_curves_dict[orig] = new

    # evaluate results
    plot_open_loris(cl_models, pretty_names_bar_dict, pretty_names_learning_curves_dict, architectures, results_dir,
                    expt_folder, file, seeds, orders, lr, save_dir=save_dir)


if __name__ == '__main__':
    # directory where all experimental results are saved
    results_dir = '/media/tyler/Data/codes/Embedded-CL/results/'

    # directory to save out plots
    save_dir = '/media/tyler/Data/codes/Embedded-CL/plots'

    main(results_dir, save_dir)
