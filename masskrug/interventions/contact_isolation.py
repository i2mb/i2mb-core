import numpy as np

from masskrug.pathogen.base_pathogen import UserStates, SymptomLevels
from .base_intervention import Intervention


class ContactIsolationIntervention(Intervention):
    def __init__(self, population, world, freeze_isolated=False, isolate_household=False,
                 dct_dropouts=0., mct_dropouts=0.):
        self.dct_dropout_ratio = dct_dropouts
        self.mct_dropout_ratio = mct_dropouts
        self.isolate_household = isolate_household
        self.population = population
        self.world = world
        self.freeze_isolated = freeze_isolated
        self.drop_outs = np.zeros((len(population), 1), dtype=bool)

        self.isolated = np.zeros((len(population), 1), dtype=bool)
        self.contact_isolated = np.zeros((len(population), 1), dtype=bool)

        # Directly isolated particles are marked as isolated when they go form incubation to infectious, observing
        # the isolation delay time.
        self.directly_isolated = np.zeros((len(population), 1), dtype=bool)

        # Number of times a particle was isolated
        self.num_isolations = np.zeros((len(population), 1), dtype=int)

        # Requests to isolate and release agents
        self.isolation_request = np.zeros((len(population), 1), dtype=bool)
        self.leave_request = np.zeros((len(population), 1), dtype=bool)

        # Isolation history, event based rendition of isolations
        self.q_history = {}

        # We keep track of isolation of non infectious particles
        self.isolated_fp = np.zeros((len(population), 1), dtype=int)
        self.isolation_time = np.zeros((len(population), 1), dtype=int)
        self.time_in_isolation = np.zeros((len(population), 1), dtype=int)
        population.add_property("isolated", self.isolated)
        population.add_property("isolation_time", self.isolation_time)
        population.add_property("time_in_isolation", self.time_in_isolation)
        population.add_property("isolated_fp", self.isolated_fp)
        population.add_property("num_isolations", self.num_isolations)
        population.add_property("isolation_request", self.isolation_request)
        population.add_property("leave_request", self.leave_request)

    def step(self, t):
        # release particles
        self.release_particles(t)

        # Remove deceased particles from the contact_isolated list.
        alive = (self.population.state != UserStates.deceased)
        self.contact_isolated &= alive
        self.isolated[~alive] = False

        self.time_in_isolation[self.isolated.ravel()] += 1
        new_isolated = self.isolation_request & ~self.isolated & alive
        self.isolation_request[:] = False

        if new_isolated.ravel().any():
            self.directly_isolated[new_isolated.ravel(), 0] = True

            # Isolate contacts
            # FIXME: This behaviour should be part of a digital contact tracing module
            contacts = set()
            if hasattr(self.population, "contact_list"):
                for cl in self.population.contact_list[new_isolated[:, 0], 0]:
                    contacts.update(cl.contacts)
                    if len(contacts) == len(self.population):
                        break

                non_contacts = ~alive | self.directly_isolated
                non_contacts = self.population.index[non_contacts.ravel()].ravel()
                contacts = contacts.difference(non_contacts)

            if self.isolate_household:
                lock_down = (self.population.home.reshape((-1, 1)) ==
                             self.population.home[list(contacts)]).any(axis=1) & ~self.directly_isolated.ravel()

                contacts.update(self.population.index[lock_down])

            # Remove drop outs added back by contacts.
            contacts = contacts.difference(list(self.drop_outs.ravel()))
            new_isolated[list(contacts), 0] = True
            dct_new_isolated = new_isolated

            # Apply drop out rates
            if hasattr(self.population, "health_authority_request"):
                mct_new_isolated = new_isolated & self.population.health_authority_request
                dct_new_isolated = new_isolated & ~self.population.health_authority_request

                if 0 < self.mct_dropout_ratio <= 1.:
                    mct_new_isolated_idx = self.population.index[mct_new_isolated.ravel()]
                    drop_outs = np.random.choice(mct_new_isolated_idx,
                                                 int(len(mct_new_isolated_idx) * self.mct_dropout_ratio),
                                                 replace=False)
                    new_isolated[drop_outs, 0] = False
                    self.drop_outs[drop_outs] = True

            if 0 < self.dct_dropout_ratio <= 1.:
                dct_new_isolated_idx = self.population.index[dct_new_isolated.ravel()]
                drop_outs = np.random.choice(dct_new_isolated_idx,
                                             int(len(dct_new_isolated_idx) * self.dct_dropout_ratio),
                                             replace=False)
                new_isolated[drop_outs, 0] = False
                self.drop_outs[drop_outs] = True

            # Compute stats
            fp = new_isolated & ~self.isolated & ~((self.population.state == UserStates.infectious) |
                                                   (self.population.state == UserStates.infected))
            self.isolated_fp[fp.ravel(), 0] += 1
            self.num_isolations[new_isolated.ravel(), 0] += ~self.isolated[new_isolated.ravel(), 0]
            self.isolated[new_isolated.ravel(), 0] = True
            self.contact_isolated[list(contacts), 0] = new_isolated[list(contacts), 0]
            self.isolation_time[new_isolated.ravel(), 0] = t

            for idx in self.population.index[new_isolated.ravel()]:
                self.q_history.setdefault(idx, []).append([t, None])

            regions = self.world.containment_region
            new_isolated = new_isolated.ravel() & (regions != self.population.location)
            for r in set(regions[new_isolated].ravel()):
                self.world.move_particles((regions == r) & new_isolated, r)

    def release_particles(self, t):
        recovered_ids = self.leave_request.ravel()
        if recovered_ids.any():
            self.population.isolated[recovered_ids, 0] = False
            self.contact_isolated[recovered_ids, 0] = False
            for idx in self.population.index[recovered_ids]:
                self.q_history[idx][-1][1] = t

            regions = self.world.home
            recovered_ids = recovered_ids.ravel() & (regions != self.population.location)
            for r in set(regions[recovered_ids]):
                self.world.move_particles((regions == r) & recovered_ids, r)

        self.leave_request[:] = False
