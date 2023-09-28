from synthetic_functions import *
import itertools
import random
from numpy.random import default_rng
import GPy


def get_data(dataset, n_train, n_query, n_test, biased_level, seed):
    if dataset == "forrester":
        biased_to_var = {0.6: 120, 0.7: 25, 0.8: 5, 0.9: 0.8}
        x, y, pairs, x_bo, y_bo = get_forrester(seed)
    elif dataset == "six_hump_camel":
        biased_to_var = {0.6: 4000, 0.7: 650, 0.8: 120, 0.9: 10}
        x, y, pairs, x_bo, y_bo = get_six_hump_camel(seed)
    elif dataset == "branin":
        biased_to_var = {0.6: 20000, 0.7: 4000, 0.8: 900, 0.9: 120}
        x, y, pairs, x_bo, y_bo = get_branin(seed)
    elif dataset == "levy":
        biased_to_var = {0.6: 180, 0.7: 40, 0.8: 10, 0.9: 2}
        x, y, pairs, x_bo, y_bo = get_levy(seed)
    else:
        raise NotImplementedError

    if biased_level == 0.5:
        y = gp_noise(x, 50)
    else:
        y = y.reshape(-1) + gp_noise(x, biased_to_var[biased_level])

    train_pairs = pairs[:n_train]
    query_pairs = pairs[n_train:n_train + n_query]
    test_pairs = pairs[n_train + n_query: n_train + n_query + n_test]

    x_duels_train = np.array(
        [[x[train_pairs[index][0]], x[train_pairs[index][1]]] for index in range(len(train_pairs))])
    pref_train = []
    for index in range(len(train_pairs)):
        pref_train.append(1) if y[train_pairs[index][0]] < y[train_pairs[index][1]] else pref_train.append(0)

    x_duels_query = np.array(
        [[x[query_pairs[index][0]], x[query_pairs[index][1]]] for index in range(len(query_pairs))])
    pref_query = []
    for index in range(len(query_pairs)):
        pref_query.append(1) if y[query_pairs[index][0]] < y[query_pairs[index][1]] else pref_query.append(0)

    x_duels_test = np.array(
        [[x[test_pairs[index][0]], x[test_pairs[index][1]]] for index in range(len(test_pairs))])
    pref_test = []
    for index in range(len(test_pairs)):
        pref_test.append(1) if y[test_pairs[index][0]] < y[test_pairs[index][1]] else pref_test.append(0)

    train_al = {'x_duels': x_duels_train, 'pref': pref_train}
    query_al = {'x_duels': x_duels_query, 'pref': pref_query}
    test_al = {'x_duels': x_duels_test, 'pref': pref_test}
    query_bo = {'x': x_bo, 'y': y_bo.reshape(-1)}
    return train_al, query_al, test_al, query_bo


def get_forrester(seed):
    rng = default_rng(seed)
    random.seed(seed)

    x = np.linspace(0, 1, 100).reshape(-1, 1)
    y = forrester(x)

    pairs = list(itertools.permutations(range(len(x)), 2))
    random.shuffle(pairs)

    x_bo = rng.uniform(0, 1, 10000).reshape(-1, 1)
    y_bo = forrester(x_bo).reshape(-1)

    return x, y, pairs, x_bo, y_bo


def get_six_hump_camel(seed):
    rng = default_rng(seed)
    random.seed(seed)

    x1 = rng.uniform(low=-3, high=3, size=1000)
    x2 = rng.uniform(low=-2, high=2, size=1000)
    x = np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1)])
    y = six_hump_camel(x1, x2)

    pairs = list(itertools.combinations(range(len(y)), 2))
    random.shuffle(pairs)

    x1_bo = rng.uniform(-2, 2, 10000)
    x2_bo = rng.uniform(-1, 1, 10000)
    x_bo = np.hstack([x1_bo.reshape(-1, 1), x2_bo.reshape(-1, 1)])
    y_bo = six_hump_camel(x1_bo, x2_bo)

    return x, y, pairs, x_bo, y_bo


def get_branin(seed):
    rng = default_rng(seed)
    random.seed(seed)

    x1 = rng.uniform(low=-5, high=10, size=1000)
    x2 = rng.uniform(low=0, high=15, size=1000)
    x = np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1)])
    y = branin(x1, x2)

    pairs = list(itertools.combinations(range(len(y)), 2))
    random.shuffle(pairs)

    x1_bo = rng.uniform(-5, 10, 10000)
    x2_bo = rng.uniform(0, 15, 10000)
    x_bo = np.hstack([x1_bo.reshape(-1, 1), x2_bo.reshape(-1, 1)])
    y_bo = branin(x1_bo, x2_bo)

    return x, y, pairs, x_bo, y_bo


def get_levy(seed):
    rng = default_rng(seed)
    random.seed(seed)

    x = rng.uniform(low=-2, high=2, size=(1000, 10))
    y = levy(x)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.shuffle(pairs)

    x_bo = rng.uniform(low=-2, high=2, size=(10000, 10))
    y_bo = levy(x_bo)

    return x, y, pairs, x_bo, y_bo


def gp_noise(x, var):
    kernel = GPy.kern.RBF(input_dim=x.shape[1], variance=var, lengthscale=0.1)
    mu = np.zeros((x.shape[0]))
    C = kernel.K(x, x)
    noise = np.random.multivariate_normal(mu, C, 1).reshape(-1)
    return noise


def logistic_function(x):
    return 1 / (1+np.e**(-x))