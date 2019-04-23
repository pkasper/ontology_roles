from sklearn.preprocessing import normalize

import numpy as np
import pandas as pd
import scipy as sp

def get_state_index(_states, _state):
    return _states.index(_state)


def create_adjacency(_sequence):

    df = pd.DataFrame.from_dict({"from": _sequence})
    df["to"] = df["from"].shift(periods=-1)

    pivot = pd.crosstab(df['from'], df['to'])
    return pivot


def calc_distribution(_states, _sequence):
    import scipy.sparse.linalg
    states = len(_states)
    full_pivot = stretch_pivot(create_adjacency(_sequence), _states)
    A = full_pivot.values

    n, m = A.shape

    num_clicks = np.sum(A)
    pr_clicks = num_clicks * 0.15 / n

    A = A + pr_clicks

    transition_matrix = sp.sparse.csr_matrix(normalize(A, axis=0, norm="l1", copy=True))
    eigval, pi = sp.sparse.linalg.eigs(transition_matrix, k=1, which="LR")
    v = pi[:, 0].real
    u = v / v.sum()
    stat_dist = transition_matrix * u
    if not np.allclose(u, stat_dist):
        print(len(_sequence))
        raise ValueError("dist not close!")
    return u, full_pivot


    # I assume that the rows of A sum to 1.
    # Therefore, In order to use A as a left multiplication matrix,
    # the transposition is necessary.
    eigvalmat = (A - scipy.sparse.eye(states)).T
    probability_distribution_constraint = scipy.ones((1, states))

    probability_distribution_constraint *= 0.15

    lhs = scipy.sparse.vstack(
        (eigvalmat,
         probability_distribution_constraint))

    B = np.zeros(states + 1)
    B[-1] = 1

    r = scipy.sparse.linalg.lsqr(lhs, B)
    # r also contains metadata about the approximation process
    p = r[0]
    return p, full_pivot


def calc_distributions_on_dict(_states, _dict):
    result = dict()

    for i, item in enumerate(_dict):
#        print("Stat dist item: " + str(i+1) + "/" + str(len(_dict)))
        result[item], pivot = calc_distribution(_states, _dict[item])
#        print(result[item])
        """
        figure, axis = plt.subplots(2, figsize=(15, 12))
        sns.heatmap(pivot, annot=True, linewidths=.5, fmt="d", ax=axis[0])
        sns.barplot(_states, result[item], ax=axis[1])
        axis_labels = axis[1].get_xticklabels()
        plt.setp(axis_labels, rotation=90)
        figure.tight_layout()
        figure.savefig("TEMP/user_sequences/" + item + ".png")
        plt.close()
        """
    return result


def prune_sequence(_sequence):
    pruned_sequence = list()
    last_item = None
    for item in _sequence:
        if item == last_item:
            continue
        else:
            pruned_sequence.append(item)
        last_item = item
    return pruned_sequence


def stretch_pivot(_pivot, _labels):
    for label_name in _labels:
        if label_name not in list(_pivot):
            _pivot[label_name] = 0

    _pivot = _pivot.transpose()
    _pivot = _pivot.sort_index()

    for label_name in _labels:
        if label_name not in list(_pivot):
            _pivot[label_name] = 0

    _pivot = _pivot.transpose()
    _pivot = _pivot.sort_index()
    _pivot = _pivot.fillna(0)

    for x in _pivot.index:
        if x not in _labels:
            _pivot.drop(x, axis=0, inplace=True)

    for x in _pivot.columns:
        if x not in _labels:
            _pivot.drop(x, axis=1, inplace=True)
    return _pivot.loc[_labels, _labels]


def pivot_epsilon_value(_pivot, _epsilon):
    _pivot = _pivot.copy()
    for c in _pivot.columns:
        for i in _pivot.index:
            if abs(_pivot.loc[c,i]) <= _epsilon:
                _pivot.loc[c,i] = np.nan
    return _pivot