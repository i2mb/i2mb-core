from itertools import combinations

import numpy as np
import scipy.spatial as ss
from scipy.special import comb

from masskrug.utils import cache_manager


def distance(a, b=None, magnitude=True):
    if b is None:
        b = a

    if magnitude:
        # return ss.distance.squareform(ss.distance.cdist(a, b, "sqeuclidean"))
        return ss.distance.pdist(a, "sqeuclidean")

    x_diff = a[:, 0, np.newaxis] - b[np.newaxis, :, 0]
    y_diff = a[:, 1, np.newaxis] - b[np.newaxis, :, 1]
    if magnitude:
        return np.sqrt(x_diff ** 2 + y_diff ** 2)

    return x_diff, y_diff


def contacts_within_radius(population, radius, return_distance=False):
    contacts = []
    n = len(population)
    if not cache_manager.is_cached(f"var_distances"):
        distances = np.full(comb(n, 2, exact=True), np.inf)
        for r in population.regions:
            if len(r.population) <= 1:
                continue

            pos = r.population.position
            d = distance(pos, pos)
            idx = r.population.index
            idx = np.array(list(combinations(idx, 2)))
            np.put(distances, ravel_index_triu_nd(idx, n), d)
            if return_distance:
                contacts.append((near_neighbours(d, radius, n, idx=idx), d[d < radius]))
            else:
                contacts.append(near_neighbours(d, radius, n, idx=idx))

            cache_manager.cache_variable(**{f"{r}_idx": idx, f"{r}_d": d})

        cache_manager.cache_variable(**{f"var_distances": distances})
    else:
        for r in population.regions:
            if len(r.population) <= 1:
                continue

            d = cache_manager.get_from_cache(f"{r}_d")
            idx = cache_manager.get_from_cache(f"{r}_idx")
            if return_distance:
                contacts.append((near_neighbours(d, radius, n, idx=idx), d[d < radius]))
            else:
                contacts.append(near_neighbours(d, radius, n, idx=idx))

    return contacts


def near_neighbours(d, radius, n, idx=None):
    candidates = d < radius
    if idx is None:
        idx = np.arange(d.shape[0])

    return idx[candidates]


def unravel_index_triu_nd(idx, n):
    if not cache_manager.is_cached(f"unraveled_index_{n}"):
        index = np.array(list(combinations(range(n), 2)))
        cache_manager.cache_variable(**{f"unraveled_index_{n}": index}, permanent=True)
    else:
        index = cache_manager.get_from_cache(f"unraveled_index_{n}")

    return index[idx]


def unravel_index_triu_nd2(idx, n):
    index = np.zeros((idx.shape[0], 2), dtype=int)
    ix_ = 0
    for i, ix in enumerate(combinations(range(n), 2)):
        if i == idx[ix_]:
            index[ix_, :] = ix
            ix_ += 1

        if ix_ == idx.shape[0]:
            break

    return index


def ravel_index_triu_nd(indices, size):
    """Ravels indices to a triangular uper matrix without diagonal. This method is compatible with scipy distance
    vectors"""
    if indices.shape[1] == 2:
        indices = indices.T

    x, y = indices

    i = np.arange(1, size, dtype=int)
    correction = (i + 1) * i // 2 + ((i + 1) % 2) * i // 2 - ((i + 1) % 2) * i // 2
    idx = y + (x * size) - np.take(correction, x)

    return idx


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


def region_ravel_multi_index(coordinates, region_index):
    """Wraps numpy ravel_multi_index function. The return value are the indices relative to the region square matrix
    rather than the entire world population matrix."""
    new_coordinates = np.zeros_like(coordinates)
    cache_position = {k: pos for k, pos in zip(region_index, range(len(region_index)))}
    for k in np.unique(coordinates):
        np.putmask(new_coordinates, coordinates == k, cache_position[k])

    shape = (len(region_index), len(region_index))

    return np.ravel_multi_index(new_coordinates, shape)
