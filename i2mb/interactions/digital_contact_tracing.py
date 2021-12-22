import numpy as np

from i2mb.utils import cache_manager
from i2mb.utils.spatial_utils import distance, contacts_within_radius
from .base_interaction import Interaction
from .contact_list import ContactList


class ContactTracing(Interaction):
    """
    Base implementation of technology based contact tracing in a particle population. This module adds the property
    `contact_list` to the population AgentList.

    :param radius: Distance between agents to consider a valid contact.
    :type radius: int
    :param population: Agent population.
    :type population: AgentList
    :param track_time: Measured in steps, track_time represents the time to consider a connection valid.
    :type track_time: int, optional
    :param duration: Measured in steps, duration is the minimal length of time required to consider recording a
     contact, defaults to 1 sample.
    :param coverage: Percentage of the population that will keep track of their contacts. Particles that will track
     coverage are selected at random.
    :param false_positives: Rate of false positives. This parameter simulates the possibility of the underlying
     technology to emmit a false positive. The model uses the area between an outer radius and the `radius`
     parameter to filter false positive agents. The outer radius is calculated using the `radius` parameter and
     the `fp_radius` parameter.
    :param false_negatives: Rate of false negatives. This parameter simulates the possibility of hte underlying
     technology failing to record a valid connection.
    :param fp_radius: Percentage of radius use to consider false positives, defaults to 0.2
    """

    def __init__(self, radius, population, track_time=None, duration=1,
                 coverage=1., false_positives=0, false_negatives=0, fp_radius=.02,
                 dropout=0., app_activation_time=0):

        self.app_activation_time = app_activation_time
        self.dropout = dropout
        self.fp_radius = fp_radius ** 2
        self.false_negatives = false_negatives
        self.false_positives = false_positives
        self.coverage = coverage
        self.radius = radius ** 2
        self.population = population
        self.track_time = track_time
        self.duration = duration

        contacts = []
        for p in population:
            contacts.append(ContactList(track_time))

        contacts = np.array(contacts).reshape((-1, 1))
        covered_particles = int(len(population) * coverage)
        mask = np.zeros(len(population), dtype=bool)
        mask[:covered_particles] = True
        np.random.shuffle(mask)
        self.covered = mask
        for i, cl in zip(mask, contacts.ravel()):
            if not i:
                cl.enabled = False

        self.contacts = contacts
        self.code = -1

        # Report of a positive test.
        self.positive_test_report = np.zeros((len(population), 1), dtype=bool)
        population.add_property("dct_positive_test_report", self.positive_test_report)

        # People contacted by digital contact tracing
        self.dct_contacted = 0

    def post_init(self):
        if hasattr(self.population, "register"):
            self.code = self.population.register("DCT")

    @cache_manager
    def distances(self):
        """Computes the distance between all agents in the population."""
        positions = self.population.position
        return distance(positions)

    def step(self, t):
        """This method is called by the :class:`i2mb.engine.core.Engine` and represent an iteration step. THe engine provides the time
         `t` of the simulation

         :param t: Simulation time, i.e., number of steps so far.
         :return: Returns the population contact list at time t
        """
        if t < self.app_activation_time:
            return

        distances = self.distances()
        fp_radius = self.radius * (1 + self.fp_radius)

        if self.false_positives > 0:
            fp_contacts = np.argwhere((distances > self.radius) & (distances <= fp_radius))
            fp_contacts = fp_contacts[np.random.choice([True, False], size=len(fp_contacts),
                                                       p=[self.false_positives, 1 - self.false_positives]), :]

            contacts = np.vstack([np.argwhere((distances <= self.radius)), fp_contacts])
        else:
            contacts = np.argwhere((distances <= self.radius))

        contacts = contacts[contacts[:, 0] != contacts[:, 1], :]
        ids = set(contacts[:, 0])
        # ids.update(fp_contacts[:, 0])
        for id_ in ids:
            contact_ids = contacts[contacts[:, 0] == id_, 1]
            if self.false_negatives > 0:
                contact_ids = contact_ids[np.random.choice([False, True], size=len(contact_ids),
                                                           p=[self.false_negatives, 1 - self.false_negatives])]
            self.contacts[id_].contact_list[0].update(contact_ids, t, self.duration)
            self.contacts[id_].contact_list[0].prune(t)

        return self.contacts

    def final(self, t):
        for cl in self.contacts.ravel():
            cl.enforce_duration(self.duration)

    def num_contacts(self):
        return np.array([len(c) for c in self.contacts.ravel()])


class RegionContactTracing(ContactTracing):
    def step(self, t):
        """This method is called by the :class:`i2mb.engine.core.Engine` and represent an iteration step. The
        engine provides the time `t` of the simulation

         :param t: Simulation time, i.e., number of steps so far.
         :return: Returns the population contact list at time t
        """
        contacts = contacts_within_radius(self.population, self.radius)

        self.dct_contacted = 0

        if self.false_negatives > 0:
            contacts = contacts[np.random.choice([False, True], size=len(contacts),
                                                 p=[self.false_negatives, 1 - self.false_negatives])]

        # Trace contacts
        for region_contacts in contacts:
            # Skip filtering  contacts when coverage is 100%
            if self.coverage < 1.:
                # Select only contacts where both ids have digital coverage.
                selector = self.covered[region_contacts.ravel()].reshape((-1, 2)).all(axis=1)
                region_contacts = region_contacts[selector, :]

            for id_a, id_b in region_contacts:
                self.contacts[id_a][0].update([id_b], t, self.duration)
                self.contacts[id_b][0].update([id_a], t, self.duration)

            for id_ in np.unique(region_contacts.ravel()):
                self.contacts[id_][0].prune(t)

        # Collect positive tests
        new_tests = self.population.test_result & self.positive_test_report
        if new_tests.any():
            self.positive_test_report[new_tests.ravel()] = False

            # Get contacts of positive tests
            contacts_idx = set()
            for cl in self.contacts[new_tests[:, 0], 0]:
                contacts_idx.update(cl.contacts)
                if len(contacts_idx) == len(self.population):
                    break

            non_contacts = self.population.index[new_tests.ravel()].ravel()
            contacts_idx = list(contacts_idx - set(non_contacts))
            contacts = np.zeros_like(new_tests)
            contacts[contacts_idx] = True

            # Apply drop out rates
            dropouts = np.random.random(contacts.shape) <= self.dropout
            contacts &= ~dropouts

            self.dct_contacted = contacts.sum()

            # Request isolation
            self.population.isolation_request[contacts.ravel()] = True
            self.population.isolated_by[contacts.ravel()] = self.code

        return
