import numpy as np

from i2mb.pathogen.base_pathogen import UserStates
from .base_intervention import Intervention


class ContactIsolationIntervention(Intervention):
    def __init__(self, population, world, relocator, freeze_isolated=False, quarantine_household=False):
        self.quarantine_household = quarantine_household
        self.population = population
        self.world = world
        self.relocator = relocator
        self.freeze_isolated = freeze_isolated

        # Currently, isolated agents
        self.isolated = np.zeros((len(population), 1), dtype=bool)

        # Number of times an agent was isolated
        self.num_confinements = np.zeros((len(population), 1), dtype=int)

        # Requests to isolate and release agents
        self.isolation_request = np.zeros((len(population), 1), dtype=bool)
        self.leave_request = np.zeros((len(population), 1), dtype=bool)

        # Isolation history, event based rendition of isolations
        self.q_history = {}

        # Modules that can request isolation and will be tracked
        self.__requesters = {}
        self.isolated_by = np.zeros((len(population), 1), dtype=int)

        # We keep track of isolation of non-infectious agents
        self.isolated_fp = np.zeros((len(population), 1), dtype=int)
        self.isolated_fn = np.zeros((len(population), 1), dtype=bool)
        self.isolation_time = np.zeros((len(population), 1), dtype=int)
        self.time_in_isolation = np.zeros((len(population), 1), dtype=int)
        self.time_in_confinement = np.zeros((len(population), 1), dtype=int)
        self.time_in_quarantine = np.zeros((len(population), 1), dtype=int)
        population.add_property("isolated", self.isolated)
        population.add_property("isolated_by", self.isolated_by)
        population.add_property("isolation_time", self.isolation_time)
        population.add_property("time_in_isolation", self.time_in_isolation)
        population.add_property("isolated_fp", self.isolated_fp)
        population.add_property("num_confinements", self.num_confinements)
        population.add_property("isolation_request", self.isolation_request)
        population.add_property("leave_request", self.leave_request)

        population.register = self.register

        # Register self when isolating households
        self.code = -1
        if quarantine_household:
            self.code = self.register("Household")

        # Counter of number of people quarantined by the household rule
        self.hh_contacted = 0

    def register(self, name):
        code = len(self.__requesters) + 1
        self.__requesters[code] = name
        return code

    def step(self, t):
        # release agents
        self.release_agents(t)

        self.hh_contacted = 0

        # Remove deceased agents from the contact_isolated list.
        alive = (self.population.state != UserStates.deceased)
        self.isolated[~alive] = False
        self.isolated_by[~alive] = 0

        self.update_false_negatives(t)

        # Updates times
        self.time_in_confinement[self.isolated.ravel()] += 1
        active = (self.population.state.ravel() > UserStates.exposed)
        self.time_in_quarantine[self.isolated.ravel() & ~active] += 1
        self.time_in_isolation[self.isolated.ravel() & active] += 1
        new_isolated = self.isolation_request & ~self.isolated & alive
        self.isolation_request[:] = False

        if new_isolated.ravel().any():
            if self.quarantine_household:
                lockdown = (self.population.home.reshape((-1, 1)) == self.population.home[new_isolated.ravel()]).any(
                    axis=1, keepdims=True) & ~new_isolated

                self.hh_contacted = lockdown.sum()

                self.isolated_by[lockdown.ravel()] = self.code
                new_isolated |= lockdown

            # Compute stats
            fp = new_isolated.ravel() & ~active
            fn = new_isolated.ravel() & active
            self.isolated_fp[fp, 0] += 1
            self.isolated_fn[fn, 0] = False
            self.num_confinements[new_isolated.ravel(), 0] += 1
            self.isolated[new_isolated.ravel(), 0] = True
            self.isolation_time[new_isolated.ravel(), 0] = t

            for idx in self.population.index[new_isolated.ravel()]:
                self.q_history.setdefault(idx, []).append([self.isolated_by[idx, 0], t, None])

            regions = self.world.containment_region
            new_isolated = new_isolated.ravel() & (regions != self.population.location)
            for r in set(regions[new_isolated].ravel()):
                self.relocator.move_agents((regions == r) & new_isolated, r)

    def release_agents(self, t):
        recovered_ids = self.leave_request.ravel()
        if recovered_ids.any():
            self.population.isolated[recovered_ids, 0] = False
            self.isolated_by[recovered_ids, 0] = 0
            for idx in self.population.index[recovered_ids]:
                self.q_history[idx][-1][2] = t

            regions = self.world.home
            recovered_ids = recovered_ids.ravel() & (regions != self.population.location)
            for r in set(regions[recovered_ids]):
                self.relocator.move_agents((regions == r) & recovered_ids, r)

            self.leave_request[:] = False

    def update_false_negatives(self, t):
        # Everyone is a false negative until isolated. This is needed for real time calculations.
        # File calculations of false negatives uses other method.
        newly_infected = self.population.time_of_infection == t
        if newly_infected.any():
            self.isolated_fn[newly_infected] = True
