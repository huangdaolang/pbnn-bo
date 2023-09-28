import numpy as np


def forrester(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def branin(x1, x2):
    a = 1
    b = 5.1 / (4 * (np.pi ** 2))
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return y


def levy(x):
    pi = np.pi

    x = 1 + (x - 1) / 4

    part1 = np.power(np.sin(pi * x[:, 0]), 2)

    part2 = np.sum(np.power(x[:, :-1] - 1, 2) * (1 + 10 * np.power(np.sin(pi * x[:, :-1] + 1), 2)), axis=1)

    part3 = np.power(x[:, -1] - 1, 2) * (1 + np.power(np.sin(2 * pi * x[:, -1]), 2))

    y = part1 + part2 + part3
    return y


def six_hump_camel(x1, x2):
    y = (4 - 2.1 * (x1 ** 2) + (x1 ** 4) / 3) * (x1 ** 2) + x1 * x2 + (-4 + 4 * (x2 ** 2)) * (x2 ** 2)
    return y
