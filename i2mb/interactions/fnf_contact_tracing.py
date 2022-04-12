import enum
from collections import Counter

import numpy as np
import networkx as nx

from i2mb.interactions.base_interaction import Interaction
from i2mb.utils.spatial_utils import contacts_within_radius


class RelationshipType(enum.IntEnum):
    stranger = -1
    friend = 0
    family = 1
    acquaintance = 2


class FriendsNFamilyContactTracing(Interaction):
    def __init__(self, network: nx.DiGraph, radius, population, track_time=7):
        self.track_time = track_time
        self.population = population
        self.radius = radius
        self.network = network

        # Tracking who reported the agent
        self.code = -1

        # Number of contacted in a series
        self._num_contacts = np.zeros((len(population), 1), dtype=int)
        self.fnf_contacted = 0

        # Report of a positive test.
        self.positive_test_report = np.zeros((len(population), 1), dtype=bool)
        population.add_property("fnf_positive_test_report", self.positive_test_report)

        self.__contacts = np.zeros((len(population), 1), dtype=bool)
        self.__weights = np.zeros((len(population), 1), dtype=bool)

        # Track encounters
        self.contact_matrix = Counter()
        self.last_update = {}

    def post_init(self, base_file_name=None):
        if hasattr(self.population, "register"):
            self.code = self.population.register("FNF")

    def step(self, t):
        self.fnf_contacted = 0

        # Keep track of last encounter
        contacts = contacts_within_radius(self.population, self.radius)
        for region_contacts in contacts:
            for r in region_contacts:
                contact_pair = tuple(r)
                if r in self.network.edges:
                    self.contact_matrix.update([contact_pair])
                    self.last_update[contact_pair] = t

        # Enforce Track time
        for k, v in self.last_update.items():
            if (t - v) > self.track_time:
                self.contact_matrix[k] = 0

        # Enforce relationship recall
        for k, duration in self.contact_matrix.items():
            recall_factor = self.network.edges[k]["recall"]
            temporal_factor = (duration / self.track_time * 2) * (1 - (t - self.last_update[
                k]) / self.track_time)
            if temporal_factor > 1:
                temporal_factor = 1

            elif temporal_factor < 0:
                temporal_factor = 0

            self.network.edges[k]["recall_probability"] = recall_factor * temporal_factor

        # Getting positive test results
        new_tests = self.population.test_result & self.positive_test_report
        if new_tests.any():
            self.positive_test_report[new_tests.ravel()] = False
            sources_idx = self.population.index[new_tests.ravel()]

            # Get positive test contacts
            contacts_idx = set()
            for idx in sources_idx:
                for idx_b in self.network.neighbors(idx):
                    k = (idx, idx_b)
                    recall = np.random.random() <= self.network.edges[k]["recall_probability"]
                    dropout = np.random.random() <= self.network.edges[k]["dropout"]
                    if recall and not dropout:
                        contacts_idx.add(idx_b)

            self.__contacts[:] = False
            self.__contacts[list(contacts_idx)] = True

            self.fnf_contacted = self.__contacts.sum()

            # Request isolation
            self.population.isolation_request[self.__contacts.ravel()] = True
            self.population.isolated_by[self.__contacts.ravel()] = self.code
