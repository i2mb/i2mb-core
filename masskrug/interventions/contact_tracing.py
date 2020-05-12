import numpy as np

from masskrug.pathogen.base_pathogen import UserStates
from .base_intervention import Intervention


class ContactTracingIntervention(Intervention):
    def __init__(self, delay, population, world):
        self.delay = delay
        self.population = population
        self.world = world

        self.isolated = np.zeros((len(population), 1), dtype=bool)
        population.add_property("isolated", self.isolated)

    def step(self, t):
        active = self.population.state == UserStates.infected
        candidates = (self.population.time_of_infection * -1 + t) >= self.delay
        isolated_ids = active & candidates
        new_isolated = isolated_ids & ~self.isolated
        recovered_ids = self.isolated & ((self.population.state == UserStates.immune) |
                                         (self.population.state == UserStates.susceptible))
        self.isolated[isolated_ids.ravel(), 0] = True

        # Isolate contacts
        contacts = set()
        if hasattr(self.population, "contact_list"):
            for cl in self.population.contact_list[new_isolated[:, 0], 0]:
                contacts.update(cl.contacts)

        new_isolated[list(contacts), 0] = True
        self.isolated[list(contacts), 0] = True
        self.isolated[recovered_ids.ravel(), 0] = False

        self.population.motion_mask[self.isolated[:, 0]] = False
        self.population.motion_mask[recovered_ids[:, 0]] = True

        self.population.position[new_isolated[:, 0]] = self.world.get_containment_positions(sum(new_isolated.ravel()))
        self.population.position[recovered_ids[:, 0]] = np.array(self.world.space_dims) / 2

        return self.isolated
