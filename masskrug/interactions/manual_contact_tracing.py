from collections import Counter

import numpy as np

from masskrug.interactions.base_interaction import Interaction
from masskrug.utils import global_time
from masskrug.utils.spatial_utils import contacts_within_radius
from masskrug.worlds import House
from masskrug.worlds.world_base import PublicSpace


class ManualContactTracing(Interaction):
    def __init__(self, radius, population, track_time=7, processing_duration=1, recall=0.4, use_restaurant_logs=False,
                 contact_network=None):

        if contact_network is None:
            contact_network = {}

        self.recall = {}
        if isinstance(recall, dict):
            self.recall.update(recall)
        else:
            self.recall[PublicSpace] = recall
            for loc in set(contact_network.values()):
                self.recall[loc] = recall

        self.use_restaurant_logs = use_restaurant_logs
        self.processing_duration = processing_duration
        self.track_time = track_time
        self.population = population
        self.radius = radius
        self.contact_matrix = Counter()
        self.last_update = {}
        self.contact_type = contact_network
        self.recall_probability = {}
        self.processing_contacts = np.zeros((len(population), 1), dtype=bool)
        self.processing_time = np.zeros((len(population), 1), dtype=int)
        self.processing_finished = np.zeros((len(population), 1), dtype=bool)
        self.contacted = np.zeros((len(population), 1), dtype=int)
        self.retried = np.zeros((len(population), 1), dtype=int)
        self.abandoned_contact = np.zeros((len(population), 1), dtype=int)
        self._num_contacts = np.zeros((len(population), 1), dtype=int)

        # Report of a positive test.
        self.positive_test_report = np.zeros((len(population), 1), dtype=bool)
        population.add_property("positive_test_report", self.positive_test_report)

    def num_contacts(self):
        return self._num_contacts

    def step(self, t):

        # Marc contacts
        contacts = contacts_within_radius(self.population, self.radius)
        for region_contacts in contacts:
            self.contact_matrix.update([tuple(r) for r in region_contacts])
            self.last_update.update({tuple(r): t for r in region_contacts})

        # Enforce Track time
        for k, v in self.last_update.items():
            if v > self.track_time:
                self.contact_matrix[k] = 0
                self._num_contacts[k, :] = 0

                # Enforce relationship recall
        for k, duration in self.contact_matrix.items():
            contact_type = self.contact_type.get(k, PublicSpace)
            recall_factor = self.recall[contact_type]
            temporal_factor = (duration / self.track_time * 2) * (1 - (t - self.last_update[
                k]) / self.track_time)
            if temporal_factor > 1:
                temporal_factor = 1

            elif temporal_factor < 0:
                temporal_factor = 0

            self.recall_probability[k] = recall_factor * temporal_factor

        # Collect positive tests
        new_tests = self.population.test_result & self.positive_test_report
        if new_tests.any():
            self.positive_test_report[new_tests.ravel()] = False
            population_index = self.population.index[new_tests.ravel()]

            # Get contacts of positive tests
            contacts = set()

            for k, v in self.contact_matrix.items():
                if v == 0:
                    continue

                idx = ~((np.array(population_index).reshape(-1, 1) == k).any(axis=0))
                if np.random.random() < self.recall_probability[k]:
                    contacts.update(np.array(k)[idx])
                    self._num_contacts[k, :] += 1

            # Mark for processing and for contacting
            contacts = list(contacts.difference(self.population.index[self.processing_contacts.ravel()]))

            self.processing_time[contacts] = t
            self.processing_contacts[contacts] = True

        ready_for_contact = (t - self.processing_time) > self.processing_duration
        ready_for_contact &= self.processing_contacts
        self.processing_contacts[ready_for_contact.ravel()] = False
        if ready_for_contact.any():
            # Apply contact error and contact for isolation and retries
            contact = np.random.random(len(self.processing_contacts)).reshape(-1, 1) < 0.8

            # setup retries
            retry = (~contact & ready_for_contact & (self.retried < 3)).ravel()
            self.processing_contacts[retry] = True
            self.processing_time[retry] = t - global_time.make_time(hour=12)
            self.retried[retry] += 1

            # log abandoned contacts Give up contacting people
            give_up_retry = (~contact & ready_for_contact & (self.retried >= 3)).ravel()
            self.abandoned_contact[give_up_retry] += 1

            # Request isolation
            contact &= ready_for_contact
            self.population.isolation_request[contact.ravel()] = True
