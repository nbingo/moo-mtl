import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pathlib import Path
import re
from multi_objective.hv import HyperVolume

from plotting import (
    font_size,
    markers,
    colors,
    natural_sort,
    lists_to_tuples,
    compare_settings,
    dirname,
    methods,
    ignore_runs,
    figsize,
    titles,
    ax_lables,
    reference_points,
    load_files,
    mean_and_std,
)

datasets = ['adult', 'compas', 'credit', 'multi_mnist', 'multi_fashion', 'multi_fashion_mnist']
methods = ['cosmos_ln']


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


p = Path(dirname)
all_files = list(p.glob('**/*.json'))

results = {}

for dataset in datasets:
    results[dataset] = {}
    for method in methods:
        # ignore folders that start with underscore
        val_file = list(sorted(p.glob(f'**/{method}/{dataset}/[!_]*/val*.json')))
        test_file = list(sorted(p.glob(f'**/{method}/{dataset}/[!_]*/test*.json')))
        train_file = list(sorted(p.glob(f'**/{method}/{dataset}/[!_]*/train*.json')))

        for r in ignore_runs:
            val_file = [f for f in val_file if str(r) not in f.parts]
            test_file = [f for f in test_file if str(r) not in f.parts]

        assert len(val_file) == len(test_file)

        data_val = load_files(val_file)
        data_test = load_files(test_file)

        if len(val_file) == 0:
            continue
        elif len(val_file) == 1:
            data_val = data_val[0]
            data_test = data_test[0]
        elif len(val_file) > 1:
            compare_settings(data_val)
            compare_settings(data_test)

        
        results[dataset][method] = {}

        s = 'start_0'
        for e in data_val[0][s]:
            if e not in data_test[0][s]:
                continue
            result_i = {}
            results[dataset][method][int(e.replace('epoch_', ''))] = {}
            if isinstance(data_val, list):
                # we have multiple runs of the same method
                
                val_scores = []
                test_scores = []
                val_hv = []
                test_hv = []
                training_time = []
                for val_run, test_run in zip(data_val, data_test):
                    
                    val_results = val_run[s][e]
                    test_results = test_run[s][e]

                    val_scores.append(val_results['scores'])
                    test_scores.append(test_results['scores'])
                    val_hv.append(val_results['hv'])
                    test_hv.append(test_results['hv'])
                    training_time.append(test_results['training_time_so_far'])
                
                result_i['val_scores'] = mean_and_std(val_scores)
                result_i['test_scores'] = mean_and_std(test_scores)
                result_i['val_hv'] = mean_and_std(val_hv)
                result_i['test_hv'] = mean_and_std(test_hv)
                result_i['training_time'] = mean_and_std(training_time)
            else:
                # we have just a single run of the method
                assert len([True for k in data_val.keys() if 'start_' in k]) == 1
                val_results = data_val[s][e]
                test_results = data_test[s][e]

                result_i['early_stop_epoch'] = int(e.replace('epoch_', ''))
                result_i['val_scores'] = val_results['scores']
                result_i['test_scores'] = test_results['scores']
                result_i['val_hv'] = val_results['hv']
                result_i['test_hv'] = test_results['hv']
                result_i['training_time'] = test_results['training_time_so_far']

            results[dataset][method][int(e.replace('epoch_', ''))] = result_i


with open('results_ablation.json', 'w') as outfile:
    json.dump(results, outfile)


# Generate the plots and tables
plt.rcParams.update({'font.size': font_size})


def plot_ablation(datasets, methods, kind, epochs=None):
    fig, axes = plt.subplots(1, len(datasets), figsize=figsize)
    for j, dataset in enumerate(datasets):
        if dataset not in results:
            continue
        ax = axes[j]
        for method in methods:
            if method not in results[dataset]:
                continue

            if kind=='convergence':
                color_shades = np.linspace(1.7, .3, len(epochs)).tolist()
                for i, e in enumerate(epochs):
                    r = results[dataset][method][e]
                    # we take the mean only
                    s = np.array(r['test_scores'][0]) if isinstance(r['test_scores'], tuple) else np.array(r['test_scores'])
                    ax.plot(
                        s[:, 0], 
                        s[:, 1], 
                        color=adjust_lightness(colors[method], amount=color_shades[i]),
                        marker=markers[method],
                        linestyle='--' if method != 'ParetoMTL' else ' ',
                        label="Epoch {}".format(e+1)
                    )

        ax.set_title(titles[dataset])
        ax.set_xlabel(ax_lables[dataset][0])
        if j==0:
            ax.set_ylabel(ax_lables[dataset][1])
        if j==2:
            ax.legend(loc='upper right')
    fig.savefig(kind + '_' + '_'.join(datasets), bbox_inches='tight')
    plt.close(fig)

datasets1 = ['multi_mnist', 'multi_fashion', 'multi_fashion_mnist']
methods1 = ['cosmos_ln']

epochs = [9, 19, 29, 39, 49]
plot_ablation(datasets1, methods1, 'convergence', epochs)
                

def plot_row(datasets, methods, limits, prefix):
    assert len(datasets) == 3
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for j, dataset in enumerate(datasets):
        if dataset not in results:
            continue
        ax = axes[j]
        for method in methods:
            if method not in results[dataset]:
                continue
            r = results[dataset][method]
            # we take the mean only
            s = np.array(r['test_scores'][0]) if isinstance(r['test_scores'], tuple) else np.array(r['test_scores'])
            if method == 'SingleTask':
                s = np.squeeze(s)
                ax.axvline(x=s[0], color=colors[method], linestyle='-.')
                ax.axhline(y=s[1], color=colors[method], linestyle='-.', label="{}".format(method))
            else:
                ax.plot(
                    s[:, 0], 
                    s[:, 1], 
                    color=colors[method],
                    marker=markers[method],
                    linestyle='--' if method != 'ParetoMTL' else ' ',
                    label="{}".format(method)
                )

                if dataset == 'multi_mnist' and method == 'cosmos_ln' and prefix == 'cosmos':
                    axins = zoomed_inset_axes(ax, 5, loc='upper right') # zoom = 6
                    axins.plot(
                        s[:, 0], 
                        s[:, 1], 
                        color=colors[method],
                        marker=markers[method],
                        linestyle='--' if method != 'ParetoMTL' else '',
                        label="{}".format(method)
                    )
                    axins.set_xlim(.26, .28)
                    axins.set_ylim(.318, .33)
                    axins.set_yticklabels([])
                    axins.set_xticklabels([])
                    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
                
                if dataset == 'multi_fashion' and method == 'cosmos_ln' and prefix == 'cosmos':
                    axins = zoomed_inset_axes(ax, 7, loc='upper right') # zoom = 6
                    axins.plot(
                        s[:, 0], 
                        s[:, 1], 
                        color=colors[method],
                        marker=markers[method],
                        linestyle='--' if method != 'ParetoMTL' else '',
                        label="{}".format(method)
                    )
                    axins.set_xlim(.4658, .4765)
                    axins.set_ylim(.503, .513)
                    axins.set_yticklabels([])
                    axins.set_xticklabels([])
                    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        ax.set_xlim(right=limits[dataset][0])
        ax.set_ylim(top=limits[dataset][1])
        ax.set_title(dataset)
        if j==2:
            ax.legend(loc='upper right')
    fig.savefig(prefix + '_' + '_'.join(datasets), bbox_inches='tight')
    plt.close(fig)







datasets = ['adult', 'compas', 'credit']
# generating the tables
header = """
\\begin{center}
\\begin{table*}[ht]
\\caption{Results on title}
\\begin{tabular}{l cc cc cc c}
\\toprule"""

column_titles1 = f"        & \multicolumn{{2}}{{c}}{{{datasets[0]}}} & \multicolumn{{2}}{{c}}{{{datasets[1]}}} & \multicolumn{{2}}{{c}}{{{datasets[2]}}} & \multirow{{2}}{{2.5cm}}{{Factor params over base model}} \\\\"
column_titles2 = """        & hv               & train time & hv               & train time & hv               & train time &     \\\\ \\midrule"""

print()