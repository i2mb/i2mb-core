import numpy as np

from masskrug.utils.spatial_utils import distance, contacts_within_radius
from masskrug.utils import cache_manager
from .base_interaction import Interaction
from .contact_list import ContactList
from .contact_matrix import ContactMatrix


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
                 coverage=1., false_positives=0, false_negatives=0, fp_radius=.02):

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
        population.add_property("contact_list", contacts)

        covered_particles = int(len(population) * coverage)
        mask = np.zeros(len(population), dtype=bool)
        mask[:covered_particles] = True
        np.random.shuffle(mask)
        for i, cl in zip(mask, contacts.ravel()):
            if not i:
                cl.enabled = False

        self.contacts = contacts

    @cache_manager
    def distances(self):
        """Computes the distance between all agents in the population."""
        positions = self.population.position
        return distance(positions)

    def step(self, t):
        """This method is called by the :class:`masskrug.engine.core.Engine` and represent an iteration step. THe engine provides the time
         `t` of the simulation

         :param t: Simulation time, i.e., number of steps so far.
         :return: Returns the population contact list at time t
        """
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
            self.population[id_].contact_list[0].update(contact_ids, t, self.duration)
            self.population[id_].contact_list[0].prune(t)

        return self.population.contact_list

    def final(self, t):
        for cl in self.contacts.ravel():
            cl.enforce_duration(self.duration)

    def num_contacts(self):
        return np.array([len(c) for c in self.contacts.ravel()])


class RegionContactTracing(ContactTracing):
    def step(self, t):
        """This method is called by the :class:`masskrug.engine.core.Engine` and represent an iteration step. THe engine provides the time
         `t` of the simulation

         :param t: Simulation time, i.e., number of steps so far.
         :return: Returns the population contact list at time t
        """
        contacts = contacts_within_radius(self.population, self.radius)

        if self.false_negatives > 0:
            contacts = contacts[np.random.choice([False, True], size=len(contacts),
                                                 p=[self.false_negatives, 1 - self.false_negatives])]

        for region_contacts in contacts:
            ids = np.unique(region_contacts[:, 0])
            for id_ in ids:
                self.population[id_].contact_list[0].update(region_contacts[region_contacts[:, 0] == id_, 1], t,
                                                            self.duration)
                self.population[id_].contact_list[0].prune(t)

            # contact list needs to be made explicitly symmetric.
            ids = np.unique(region_contacts[:, 1])
            for id_ in ids:
                self.population[id_].contact_list[0].update(region_contacts[region_contacts[:, 1] == id_, 0], t,
                                                            self.duration)
                self.population[id_].contact_list[0].prune(t)

        return self.population.contact_list
