import numpy as np


def callable_number(number):
    def __callable(size=None):
        if size is None:
            return number

        else:
            return np.full(size, number)
    return __callable
