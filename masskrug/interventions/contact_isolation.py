import numpy as np

from masskrug.pathogen.base_pathogen import UserStates
from .base_intervention import Intervention


class ContactIsolationIntervention(Intervention):
    def __init__(self, delay, population, world, quarantine_duration=5, test_to_exit=True, test_duration=0,
                 freeze_isolated=False, isolate_household=False):
        self.isolate_household = isolate_household
        self.test_duration = test_duration
        self.test_to_exit = test_to_exit
        self.delay = delay
        self.population = population
        self.world = world
        self.quarantine = quarantine_duration
        self.freeze_isolated = freeze_isolated

        self.isolated = np.zeros((len(population), 1), dtype=bool)
        self.contact_isolated = np.zeros((len(population), 1), dtype=bool)

        # Directly isolated particles are marked as isolated when they go form incubation to infectious, observing
        # the isolation delay time.
        self.directly_isolated = np.zeros((len(population), 1), dtype=bool)

        # Number of times a particle was isolated
        self.num_isolations = np.zeros((len(population), 1), dtype=int)

        # We keep track of isolation of non infectious particles
        self.isolated_fp = np.zeros((len(population), 1), dtype=int)
        self.isolation_time = np.zeros((len(population), 1), dtype=int)
        self.time_in_isolation = np.zeros((len(population), 1), dtype=int)
        population.add_property("isolated", self.isolated)
        population.add_property("isolation_time", self.isolation_time)
        population.add_property("time_in_isolation", self.time_in_isolation)
        population.add_property("isolated_fp", self.isolated_fp)
        population.add_property("num_isolations", self.num_isolations)

    def step(self, t):
        # Remove deceased particles from the contact_isolated list.
        alive = (self.population.state != UserStates.deceased)
        self.contact_isolated &= alive
        self.isolated[self.population.state == UserStates.deceased] = False

        active = self.population.state == UserStates.infected
        candidates = (t - (self.population.time_of_infection + self.population.incubation_period)) >= self.delay
        isolated_ids = active & candidates
        new_isolated = isolated_ids & (~self.isolated | (self.isolated & self.contact_isolated))

        if self.test_to_exit:
            recovered_ids = self.isolated & (((self.population.state == UserStates.immune) |
                                              (self.population.state == UserStates.susceptible)) &
                                             (t - self.isolation_time > self.test_duration))
        else:
            recovered_ids = self.isolated & (((self.population.state == UserStates.immune) |
                                              (self.population.state == UserStates.asymptomatic) |
                                              (self.population.state == UserStates.susceptible)) &
                                             ((t - self.isolation_time) > self.quarantine))

        if self.isolate_household:
            affected = ~recovered_ids * self.isolated
            lock_down = (self.population.home.reshape((-1, 1)) == self.population.home[affected.ravel()]).any(axis=1)
            recovered_ids &= ~lock_down.reshape((-1, 1))

        self.time_in_isolation[self.isolated.ravel()] += 1

        if new_isolated.ravel().any():
            self.num_isolations[new_isolated.ravel(), 0] += ~self.isolated[new_isolated.ravel(), 0]
            self.isolated[new_isolated.ravel(), 0] = True
            self.directly_isolated[new_isolated.ravel(), 0] = True
            self.contact_isolated[new_isolated.ravel(), 0] = False
            self.isolation_time[new_isolated.ravel(), 0] = t

            # Isolate contacts
            contacts = set()
            if hasattr(self.population, "contact_list"):
                for cl in self.population.contact_list[new_isolated[:, 0], 0]:
                    contacts.update(cl.contacts)
                    if len(contacts) == len(self.population):
                        break

                non_contacts = ~alive | self.directly_isolated
                non_contacts = self.population.index[non_contacts.ravel()].ravel()
                contacts = contacts.difference(non_contacts)

            else:
                # Lock down the entire location
                lock_down = ((self.population.location.reshape((-1, 1)) ==
                              self.population.location[new_isolated.ravel()]).any(axis=1) &
                             ~self.directly_isolated.ravel())
                contacts.update(self.population.index[lock_down])

            if self.isolate_household:
                lock_down = (self.population.home.reshape((-1, 1)) ==
                             self.population.home[list(contacts)]).any(axis=1) & ~self.directly_isolated.ravel()

                contacts.update(self.population.index[lock_down])

            new_isolated[list(contacts), 0] = True
            fp = new_isolated & ~self.isolated & ~((self.population.state == UserStates.asymptomatic) |
                                                   (self.population.state == UserStates.incubation) |
                                                   (self.population.state == UserStates.infected))
            self.isolated_fp[fp.ravel(), 0] += 1
            self.num_isolations[list(contacts), 0] += ~self.isolated[list(contacts), 0]
            self.isolated[list(contacts), 0] = True
            self.contact_isolated[list(contacts), 0] = True
            self.isolation_time[list(contacts), 0] = t

            if self.freeze_isolated:
                self.population.motion_mask[new_isolated.ravel()] = False

            regions = self.world.containment_region
            new_isolated = new_isolated.ravel() & (regions != self.population.location)
            for r in set(regions[new_isolated].ravel()):
                self.world.move_particles((regions == r) & new_isolated, r)

        if recovered_ids.ravel().any():
            self.isolated[recovered_ids.ravel(), 0] = False
            self.contact_isolated[recovered_ids.ravel(), 0] = False
            if self.freeze_isolated:
                self.population.motion_mask[recovered_ids.ravel()] = True

            regions = self.world.home
            recovered_ids = recovered_ids.ravel() & (regions != self.population.location)
            for r in set(regions[recovered_ids]):
                self.world.move_particles((regions == r) & recovered_ids, r)

        return self.isolated
