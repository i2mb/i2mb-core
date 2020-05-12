import numpy as np


def distance(a, b=None, magnitude=True):
    if b is None:
        b = a

    x_diff = a[:, 0, np.newaxis] - b[np.newaxis, :, 0]
    y_diff = a[:, 1, np.newaxis] - b[np.newaxis, :, 1]
    if magnitude:
        return np.sqrt(x_diff ** 2 + y_diff ** 2)

    return x_diff, y_diff


def sum_field(a, b):
    a_x = a[:, 0, np.newaxis]
    a_y = a[:, 1, np.newaxis]
    b_x = b[np.newaxis, :, 0]
    b_y = b[np.newaxis, :, 1]
    g_x = (a_x + b_x).sum(axis=1)
    g_y = (a_y + b_y).sum(axis=1)

    res = np.zeros(a.shape)
    res[:, 0] = g_x
    res[:, 1] = g_y
    return res
