import seaborn as sns
from seaborn.external import husl

from matplotlib.colors import ListedColormap

import random

s = 90
l = 65
h = 360

n = 10

shuffle_seed_numbers = list(range(n))
random.shuffle(shuffle_seed_numbers)
shuffle_seed_numbers = [1, 3, 7, 8, 2, 4, 5, 0, 6, 9]
shuffle_seed = [h/n * x for x in shuffle_seed_numbers if x < n]


def get_static(_i):
    _i = min(_i, len(shuffle_seed) - 1)
    if _i < 0:
        return 0, 0, 0
    return husl.husl_to_rgb(shuffle_seed[_i], s, l)


def get_full_map():
    return ListedColormap(sns.color_palette([get_static(x) for x in range(n)]))


def get_gradient(_i):
    _i = min(_i, len(shuffle_seed) - 1)
    return sns.light_palette((shuffle_seed[_i], s, l), as_cmap=True, input="husl")


def get_diverging(_i, _j):
    _i = min(_i, len(shuffle_seed) - 1)
    _j = min(_j, len(shuffle_seed) - 1)
    return sns.diverging_palette(shuffle_seed[_i], shuffle_seed[_j], s=s, l=l, as_cmap=True)


def get_static_plotly(_i, alpha=1):
    return "rgba({colorstring}, {alpha})".format(colorstring=",".join([str(int(x*255)) for x in get_static(_i)]), alpha=alpha)