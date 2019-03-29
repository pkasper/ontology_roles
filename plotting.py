import matplotlib
import mpl_toolkits
matplotlib.rc('pdf', fonttype=42)

from matplotlib import pyplot as plt

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
sns.set(style="ticks")

import colormap
import lib

import sys
import os
import math
import pickle
from tabulate import tabulate
import pandas as pd

from configparser import ConfigParser
cfg = ConfigParser()
cfg.read("config.cfg")


def transition_matrix(_figsize, _event_count, _pivot, _transition_count_pivot, _c_index, _filename, logscale=False):
    _pivot = lib.pivot_epsilon_value(_pivot, 0.001)
    sns.set(style="ticks", font_scale=4, rc={"xtick.major.size": 20, "xtick.major.width": 5, "ytick.major.size": 20, "ytick.major.width": 5})
    figure, axis = plt.subplots(2, 2, sharex="col", figsize=_figsize, gridspec_kw={'height_ratios': [1, 2], 'width_ratios': [15, 1]})
    
    if logscale:
        _event_count = np.array(_event_count) + 1 # avoid log(0)
    axis[0][0].bar(np.arange(len(_event_count)), _event_count, color=colormap.get_static(_c_index), align='edge')
    axis[0][0].set_ylabel("# Edit Actions")
    axis[0][0].set_yscale("log")
    #axis[0][0].set_ylim(0, 50000)
    axis[0][1].remove()

    cbar_sections = np.linspace(0, 1, 11)
    sns.heatmap(_pivot, ax=axis[1][0], linewidths=.5, cbar=True, cbar_ax=axis[1][1], cmap=colormap.get_gradient(_c_index), cbar_kws={"ticks": cbar_sections, "boundaries": cbar_sections})

    """
    for label in axis[1][0].get_xticklabels():
        label.set_text(LABEL_NAMES(label.get_text()))
    for label in axis[1][0].get_yticklabels():
        label.set_text(LABEL_NAMES(label.get_text()))
    """
    axis[1][0].set_xticklabels(axis[1][0].get_xticklabels())
    axis[1][0].set_yticklabels(axis[1][0].get_yticklabels())

    sns.despine(ax=axis[0][0], top=True, bottom=False, left=False, right=True, trim=True)
    axis[1][0].set_xlabel("To Action")
    axis[1][0].set_ylabel("From Action")
    figure.tight_layout()
    figure.savefig(_filename + ".png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + ".pdf", transparent=True, bbox_inches="tight")
    print(_filename + ".png")
    print(_filename + ".pdf")
    plt.close("all")


def k_means(_stat_dist_reduced, _num_kernels, _labels, _sample_silhouette_values, _silhouette_avg, _filename):
    sns.set(style="ticks", font_scale=4, rc={"xtick.major.size": 20, "xtick.major.width": 5, "ytick.major.size": 20, "ytick.major.width": 5})
    figure, axis = plt.subplots(figsize=(25, 20))
#    figure = plt.figure(figsize=(30, 10))
#    axis = figure.add_subplot(131)
    y_lower = 10
    for i in range(_num_kernels):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = _sample_silhouette_values[_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / _num_kernels)
        axis.fill_betweenx(np.arange(y_lower, y_upper),
                           0, ith_cluster_silhouette_values,
                           facecolor=colormap.get_static(i), edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        #axis.text(0.05, y_lower + 0.5 * size_cluster_i, EDITOR_TYPES(i))  # , color=colormap.get_static(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    axis.set_xlabel("Silhouette Coefficient Values")
    axis.set_ylabel("Cluster Label")

    # The vertical line for average silhouette score of all the values
    axis.axvline(x=_silhouette_avg, ymin=0.03, ymax=0.97, color="red", linestyle="--")
    axis.text(_silhouette_avg, y_lower + 20, "avg: " + str(round(_silhouette_avg, 5)), color="red", horizontalalignment="center")

    axis.set_yticks([])  # Clear the yaxis labels / ticks
    axis.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    sns.despine(ax=axis, top=True, bottom=False, left=False, right=True, trim=True)
    figure.savefig(_filename + "_silhouette.png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + "_silhouette.pdf", transparent=True, bbox_inches="tight")
    print(_filename + "_silhouette.png")
    print(_filename + "_silhouette.pdf")

    figure = plt.figure(figsize=(25, 20))
    axis = figure.add_subplot(111, projection='3d')
    axis.scatter([x[0] for x in _stat_dist_reduced],
                 [x[1] for x in _stat_dist_reduced],
                 zs=[x[2] for x in _stat_dist_reduced],
                 c=[colormap.get_static(x) for x in _labels], s=100)
    figure.savefig(_filename + "_clusters.png", transparent=True, bbox_inches="tight")
    #figure.savefig(_filename + "_clusters.pdf", transparent=True, bbox_inches="tight")
    with open(_filename + "_clusters.p", "wb") as pickle_file:
        dump_data = {
            "x": [x[0] for x in _stat_dist_reduced],
            "y": [x[1] for x in _stat_dist_reduced],
            "z": [x[2] for x in _stat_dist_reduced],
            "c": [colormap.get_static(x) for x in _labels]
        }
        pickle.dump(dump_data, pickle_file)
    print(_filename + "_clusters.png")
    print(_filename + "_clusters.pdf")
    
    
    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    axes[0].scatter([x[0] for x in _stat_dist_reduced],[x[1] for x in _stat_dist_reduced], c=[colormap.get_static(x) for x in _labels], s=100)
    axes[1].scatter([x[0] for x in _stat_dist_reduced],[x[2] for x in _stat_dist_reduced], c=[colormap.get_static(x) for x in _labels], s=100)
    axes[2].scatter([x[1] for x in _stat_dist_reduced],[x[2] for x in _stat_dist_reduced], c=[colormap.get_static(x) for x in _labels], s=100)
    figure.tight_layout()
    figure.savefig(_filename + "_clusters_planar.png", transparent=True, bbox_inches="tight")
    #figure.savefig(_filename + "_clusters_planar.pdf", transparent=True, bbox_inches="tight")
    print(_filename + "_clusters_planar.png")
    print(_filename + "_clusters_planar.pdf")
    

    figure, axis = plt.subplots(figsize=(25, 20))
    # axis = figure.add_subplot(133)
    cluster_size = defaultdict(int)
    for i, c in enumerate(_labels):
        cluster_size[c] += 1

    axis.bar(list(cluster_size.keys()), list(cluster_size.values()),
             color=[colormap.get_static(x) for x in list(cluster_size.keys())])
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Population Size")
    sns.despine(ax=axis, top=True, bottom=False, left=False, right=True, trim=True)

    figure.savefig(_filename + "_population.png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + "_population.pdf", transparent=True, bbox_inches="tight")
    print(_filename + "_population.png")
    print(_filename + "_population.pdf")
    plt.close("all")


def usertype_comparison(_a, _b, _filename):
    sns.set(style="ticks", font_scale=4, rc={"xtick.major.size": 20, "xtick.major.width": 5, "ytick.major.size": 20, "ytick.major.width": 5})
    all_types = sorted(list(set(_a.keys()).union(_b.keys())))
    figure, axis = plt.subplots(figsize=(25, 20))
    a_types = np.asarray([_a[t] if t in _a else 0 for t in all_types])
    b_types = np.asarray([_b[t] if t in _b else 0 for t in all_types])
    axis.bar(np.arange(len(all_types)), a_types, facecolor=colormap.project_a)
    axis.bar(np.arange(len(all_types)), b_types, bottom=a_types, facecolor=colormap.project_b)

    axis.set_ylim(0, max(a_types + b_types) + 5 - (max(a_types + b_types) % 5))

    axis.set_xticks(np.arange(len(all_types)))
    axis.set_xticklabels([EDITOR_TYPES(x) for x in all_types], rotation=45)
    [l.set_color(colormap.get_static(i)) for i, l in enumerate(axis.get_xticklabels())]
    sns.despine(ax=axis, top=True, bottom=False, left=False, right=True, trim=True)
    axis.set_xlabel("Editor Type")
    axis.set_ylabel("# Users")
    figure.tight_layout()
    figure.savefig(_filename + ".png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + ".pdf", transparent=True, bbox_inches="tight")
    print(_filename + ".png")
    print(_filename + ".pdf")
    plt.close("all")


def usertype_distributions(_df, _filename):
    sns.set(style="ticks", font_scale=6, rc={"xtick.major.size": 20, "xtick.major.width": 5, "ytick.major.size": 20, "ytick.major.width": 5})
    figure, axis = plt.subplots(figsize=(20, 15))
    indices = np.arange(len(_df))
    for i, col in enumerate(_df.columns):
        bottom = np.zeros(len(_df))
        for j, x in enumerate(_df.columns):
            if j >= i:
                break
            bottom += _df[x]
        axis.bar(indices, _df[col], 1, bottom=bottom, color=colormap.get_static(col), label=EDITOR_TYPES(col))
    axis.xaxis.set_ticklabels([])

    axis.set_xlim(0, len(_df))
    axis.set_yscale('log')

    axis.set_xlabel("Ontology Projects")
    axis.set_ylabel("Edit-Actions")

    sns.despine(ax=axis, top=True, bottom=False, left=False, right=True, trim=True)
    figure.tight_layout()
    figure.savefig(_filename + ".png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + ".pdf", transparent=True, bbox_inches="tight")
    print(_filename + ".png")
    print(_filename + ".pdf")



    # NORMALIZED
    df_norm = _df.div(_df.sum(axis=1), axis=0).sort_values(by=[i for i in range(len(_df.columns))], ascending=[0 for i in range(len(_df.columns))])

    figure, axis = plt.subplots(figsize=(20, 15))
    indices = np.arange(len(df_norm))
    for i, col in enumerate(df_norm.columns):
        bottom = np.zeros(len(df_norm))
        for j, x in enumerate(df_norm.columns):
            if j >= i:
                break
            bottom += df_norm[x]
        axis.bar(indices, df_norm[col], 1, bottom=bottom, color=colormap.get_static(col), label=EDITOR_TYPES(col))
    axis.xaxis.set_ticklabels([])

    axis.set_xlim(0, len(df_norm))

    axis.set_xlabel("Ontology Projects")
    axis.set_ylabel("User Type Fraction")

    sns.despine(ax=axis, top=True, bottom=False, left=False, right=True, trim=True)
    figure.tight_layout()
    figure.savefig(_filename + "_normalized.png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + "_normalized.pdf", transparent=True, bbox_inches="tight")
    print(_filename + "_normalized.png")
    print(_filename + "_normalized.pdf")

    figlegend = plt.figure(figsize=(12, 0.5))
    axis_labels, axis_handles = axis.get_legend_handles_labels()
    figlegend.legend(axis_labels, axis_handles, ncol=5, fontsize=15, loc='upper center')
    figlegend.savefig(_filename + "_legend.png", transparent=True, bbox_inches="tight")
    figlegend.savefig(_filename + "_legend.pdf", transparent=True, bbox_inches="tight")
    print(_filename + "_legend.png")
    print(_filename + "_legend.pdf")
    plt.close("all")


def project_scatter(_fractions, _counts, pca_fit, _scale_function, _filename):
    sns.set(style="ticks", font_scale=6, rc={"xtick.major.size": 20, "xtick.major.width": 5, "ytick.major.size": 20, "ytick.major.width": 5})
    figure, axis = plt.subplots(figsize=(20, 15))
    for i, row in enumerate(_fractions.iterrows()):
        sections = []
        carry = 0
        for section in _fractions.columns:
            if row[1][section] == 0:
                sections.append(None)
                continue
            x = [0] + np.cos(np.linspace(2*math.pi*carry, 2 * math.pi * (carry + row[1][section]), 100)).tolist()
            y = [0] + np.sin(np.linspace(2*math.pi*carry, 2 * math.pi * (carry + row[1][section]), 100)).tolist()
            sections.append(list(zip(x, y)))
            carry += row[1][section]
        for j, section in enumerate(sections):
            if section is not None:
                axis.scatter(pca_fit[i][0], pca_fit[i][1], s=_scale_function(_counts[i]), facecolor=colormap.get_static(j), marker=(section, 0))
    sns.despine(ax=axis, top=True, bottom=False, left=False, right=True, trim=True)
    figure.savefig(_filename + ".png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + ".pdf", transparent=True, bbox_inches="tight")
    print(_filename + ".png")
    print(_filename + ".pdf")
    plt.close("all")


def project_scatter_3d(_fractions, _counts, pca_fit, _scale_function, _filename):
    sns.set(style="ticks", font_scale=6, rc={"xtick.major.size": 20, "xtick.major.width": 5, "ytick.major.size": 20, "ytick.major.width": 5})
    figure = plt.figure(figsize=(20, 15))
    axis = figure.add_subplot(111, projection="3d")
    for i, row in enumerate(_fractions.iterrows()):
        sections = []
        carry = 0
        for section in _fractions.columns:
            if row[1][section] == 0:
                sections.append(None)
                continue
            x = [0] + np.cos(np.linspace(2*math.pi*carry, 2 * math.pi * (carry + row[1][section]), 100)).tolist()
            y = [0] + np.sin(np.linspace(2*math.pi*carry, 2 * math.pi * (carry + row[1][section]), 100)).tolist()
            sections.append(list(zip(x, y)))
            carry += row[1][section]
        for j, section in enumerate(sections):
            if section is not None:
                axis.scatter(pca_fit[i][0], pca_fit[i][1], zs=pca_fit[i][2], s=_scale_function(_counts[i]), facecolor=colormap.get_static(j), marker=(section, 0))
    figure.savefig(_filename + ".png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + ".pdf", transparent=True, bbox_inches="tight")
    print(_filename + ".png")
    print(_filename + ".pdf")
    plt.close("all")


def aics_bics(_data, _filename):
    figure, axes = plt.subplots(len(_data), figsize=(20, 15))

    for axis_index, label_level in enumerate(_data):

        plot_data = {"x": [], "y": []}
        for x, y in sorted(_data[label_level]["aics"].items()):
            if x >= len(_data[label_level]["aics"]) - 3:
                continue
            plot_data['x'].append(x)
            plot_data['y'].append(y)
        plot_data['x'] = np.asarray(plot_data['x'])
        plot_data['y'] = np.asarray(plot_data['y'])
        plot_data['y'] = plot_data['y'] - plot_data['y'].max()
        axes[axis_index].plot(plot_data['x'],
                              plot_data['y'],
                              label="AICS")
        axes[axis_index].set_title("Label Level: " + str(label_level))
        axes[axis_index].set_yscale("symlog")

        right_axis = axes[axis_index].twinx()

        plot_data = {"x": [], "y": []}
        for x, y in sorted(_data[label_level]["bics"].items()):
            if x >= len(_data[label_level]["bics"]) - 3:
                continue
            plot_data['x'].append(x)
            plot_data['y'].append(y)
        plot_data['x'] = np.asarray(plot_data['x'])
        plot_data['y'] = np.asarray(plot_data['y'])
        plot_data['y'] = plot_data['y'] - plot_data['y'].max()
        right_axis.plot(plot_data['x'],
                        plot_data['y'],
                        label="BICS")
        right_axis.set_yscale("symlog")

        sns.despine(ax=axes[axis_index], top=True, bottom=False, left=False, right=False, trim=True)
        axes[axis_index].legend()
        print("label level", label_level)
    figure.savefig(_filename + ".png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + ".pdf", transparent=True, bbox_inches="tight")
    print(_filename + ".png")
    print(_filename + ".pdf")
    plt.close("all")


def centroids_bar_plot(_centroids, _dimension_labels, _centroid_order, _filename):

    bar_width = 1/(len(_centroids) + 2)
    overall_min = min([min(c['values']) for c in _centroids])
    overall_max = max([max(c['values']) for c in _centroids])
    figure, axis = plt.subplots(figsize=(10, 4))
    for centroid_index, centroid in enumerate(_centroids):
        axis.bar(np.arange(len(centroid['values'])) - 0.3 + bar_width * centroid_index, centroid['values'], width=bar_width, label=centroid['label'], color=colormap.get_static(_centroid_order[centroid_index]))

    axis.set_xticks(np.arange(len(_dimension_labels)), minor=False)
    for label_index, label in enumerate(axis.get_xticklabels()):
        label.set_text(_dimension_labels[label_index])

    #axis.set_yticks([])

    axis.set_xticklabels(axis.get_xticklabels(), rotation="45")
    #axis.set_ylim(overall_min, overall_max)

    axis.legend(title="Centroid")

    axis.set_yscale("log")

    axis.set_ylabel("$log$ (Edit Action Probability)")
    axis.set_xlabel("Labels")

    figure.tight_layout()
    sns.despine(ax=axis, top=True, bottom=False, left=False, right=True, trim=True)

    figure.savefig(_filename + ".png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + ".pdf", transparent=True, bbox_inches="tight")
    print(_filename + ".png")
    print(_filename + ".pdf")
    plt.close("all")


def dominance(_df, _filename):
    line_width = 5.0
    matplotlib.rcParams['hatch.linewidth'] = line_width * 1.5
    # dominance distribution
    sns.set(style="ticks", font_scale=6,
            rc={"xtick.major.size": 20, "xtick.major.width": 5, "ytick.major.size": 20, "ytick.major.width": 5})
    figure, axis = plt.subplots(figsize=(25, 10))

    test = np.zeros((len(_df), len(_df.columns)))

    c = 0
    for _, changes in _df.iterrows():
        test[c] = sorted(list(changes), reverse=True)
        test[c] /= test[c].sum()
        c += 1

    df_new = pd.DataFrame(test)
    df_new = df_new.sort_values(list(range(len(_df.columns))), ascending=False)
    print(tabulate(df_new))
    df_new = df_new.cumsum(axis=1)

    hatches = ["/","\\","x",".",".","*"]
    labels = ["$1^{st}$", "$2^{nd}$", "$3^{rd}$", "$4^{th}$", "$5^{th}$"]
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    label_handles = []
    for lead_index in range(len(_df.columns)):

        if lead_index > 0:
            bottom = df_new[lead_index - 1]
        else:
            bottom = np.zeros(len(df_new))
        if df_new[lead_index].sum() == bottom.sum():
            print("no entries for: ", lead_index)
            continue

        axis.fill_between(np.arange(len(test)), bottom, df_new[lead_index], color=flatui[lead_index]+"0F" , label=labels[lead_index])
        label_handles.append(axis.fill_between(np.arange(len(test)), bottom, df_new[lead_index], hatch=hatches[lead_index], edgecolor=flatui[lead_index], facecolor="#FFFFFF88" , label=labels[lead_index]))
        axis.plot(np.arange(len(df_new)), df_new[lead_index], linewidth=line_width, color=flatui[lead_index])
        print(lead_index)

    axis.set_ylabel("Fractions of changes\nby leading roles")
    axis.set_xlabel("Projects")
    axis.set_xticklabels([])
    axis.legend(handles=label_handles, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)

    sns.despine(ax=axis, top=True, bottom=False, left=False, right=True, trim=True)
    figure.tight_layout()
    figure.savefig(_filename + "_total.png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + "_total.pdf", transparent=True, bbox_inches="tight")
    print(_filename + "_total.png")
    print(_filename + "_total.pdf")

    return
    # Projects
    figure, axis = plt.subplots(figsize=(25, 10))

    names = []
    values = []
    colors = []
    for i in range(_df['dominant_cluster'].min(), _df['dominant_cluster'].max() + 1):
        print(i)
        names.append(EDITOR_TYPES(i))
        values.append(sum(_df['dominant_cluster'] == i))
        colors.append(colormap.get_static(i))
    axis.bar(np.arange(len(values)), values, color=colors, edgecolor="None",  alpha = 0.5)
    axis.bar(np.arange(len(values)), values, facecolor="None", edgecolor=colors, linewidth=5)
    axis.set_xlabel("Editor Role")
    axis.set_ylabel("# Projects")
    axis.set_xticklabels(names)
    sns.despine(ax=axis, top=True, bottom=False, left=False, right=True, trim=True)
    figure.tight_layout()
    figure.savefig(_filename + "_clusters.png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + "_clusters.pdf", transparent=True, bbox_inches="tight")
    print(_filename + "_clusters.png")
    print(_filename + "_clusters.pdf")
    
def aics_bics(_data, _filename):
    figure, axes = plt.subplots(len(_data), figsize=(20, 15))

    for axis_index, label_level in enumerate(_data):

        plot_data = {"x": [], "y": []}
        for x, y in sorted(_data[label_level]["aics"].items()):
            if x >= len(_data[label_level]["aics"]) - 3:
                continue
            plot_data['x'].append(x)
            plot_data['y'].append(y)
        plot_data['x'] = np.asarray(plot_data['x'])
        plot_data['y'] = np.asarray(plot_data['y'])
        plot_data['y'] = plot_data['y'] - plot_data['y'].max()
        axes[axis_index].plot(plot_data['x'],
                              plot_data['y'],
                              label="AICS")
        axes[axis_index].set_title("Label Level: " + str(label_level))
        axes[axis_index].set_yscale("symlog")

        right_axis = axes[axis_index].twinx()

        plot_data = {"x": [], "y": []}
        for x, y in sorted(_data[label_level]["bics"].items()):
            if x >= len(_data[label_level]["bics"]) - 3:
                continue
            plot_data['x'].append(x)
            plot_data['y'].append(y)
        plot_data['x'] = np.asarray(plot_data['x'])
        plot_data['y'] = np.asarray(plot_data['y'])
        plot_data['y'] = plot_data['y'] - plot_data['y'].max()
        right_axis.plot(plot_data['x'],
                        plot_data['y'],
                        label="BICS")
        right_axis.set_yscale("symlog")

        sns.despine(ax=axes[axis_index], top=True, bottom=False, left=False, right=False, trim=True)
        axes[axis_index].legend()
        print("label level", label_level)
    print("FIG CREATED")
    figure.savefig(_filename + ".png", transparent=True, bbox_inches="tight")
    figure.savefig(_filename + ".pdf", transparent=True, bbox_inches="tight")
    print(_filename + ".png")
    print(_filename + ".pdf")
    plt.close("all")