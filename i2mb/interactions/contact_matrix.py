from itertools import combinations

import numpy as np

from i2mb.utils.spatial_utils import ravel_index_triu_nd


class ContactMatrix:
    def __init__(self, n):
        self.__index = np.array(list(combinations(range(n), 2)))
        self.__contacts = np.zeros(self.__index.shape[0], dtype=bool)
        self.__contact_duration = np.zeros(self.__index.shape[0], dtype=int)
        self.n = n

    def update_contacts(self, contacts):
        idx = ravel_index_triu_nd(contacts, self.n)
        self.__contacts[idx] = True
        self.__contact_duration[idx] += 1
        return

    def get_sufficient_contact(self, duration):
        return self.__index[self.__contact_duration >= duration]

    def reset(self):
        self.__contact_duration[~self.__contacts] = 0
        self.__contacts[:] = False
