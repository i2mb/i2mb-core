from functools import partial

import numpy as np

from masskrug.interactions.contact_list import ContactList
from masskrug.engine.particle import ParticleList
from masskrug.utils.spatial_utils import distance
from masskrug.utils import cache_manager
from .base_pathogen import Pathogen, SymptomLevels, UserStates


class CoronaVirus(Pathogen):
    def __init__(self, radius, exposure_time, population: ParticleList,
                 duration_distribution=None,
                 asymptomatic_p=0.01,
                 death_rate=None,
                 icu_beds=None):

        self.radius = radius
        self.exposure_time = exposure_time
        self.population = population
        self.asymptomatic_p = asymptomatic_p
        self.icu_beds = icu_beds

        self.duration_distribution = duration_distribution
        if duration_distribution is None:
            self.duration_distribution = partial(np.random.normal, 14 * (3600 * 24), 3 * (3600 * 24))

        try:
            self.__death_rate, self.__death_rate_icu = death_rate
        except TypeError:
            self.__death_rate, self.__death_rate_icu = death_rate, death_rate

        if self.__death_rate is None:
            self.__death_rate = .01

        if self.__death_rate is None:
            self.__death_rate = .2

        self.death_rate = self.__death_rate

        shape = (len(population), 1)
        self.states = np.ones(shape) * UserStates.susceptible
        self.symptom_levels = np.ones(shape) * SymptomLevels.not_sick
        self.duration_infection = np.zeros(shape)
        self.time_of_infection = np.zeros(shape)
        self.particles_infected = np.zeros(shape)
        self.contacts = []
        for p in population:
            self.contacts.append(ContactList())

        population.add_property("state", self.states)
        population.add_property("symptom_level", self.symptom_levels)
        population.add_property("duration_infection", self.duration_infection)
        population.add_property("time_of_infection", self.time_of_infection)
        population.add_property("particles_infected", self.particles_infected)

        self.outcome = {}

    def introduce_pathogen(self, num_p0s, t, asymptomatic=None):
        state = UserStates.infected
        if isinstance(asymptomatic, float):
            if asymptomatic <= 1:
                state = np.random.choice([UserStates.infected, UserStates.asymptomatic], num_p0s,
                                         p=[1 - asymptomatic, asymptomatic])
            else:
                # We understand numbers greater than one as the number of asymptomatic particles.
                asymptomatic = int(asymptomatic)

        elif isinstance(asymptomatic, bool):
            if asymptomatic:
                state = np.random.choice([UserStates.infected, UserStates.asymptomatic], num_p0s,
                                         p=[1 - asymptomatic, self.asymptomatic_p])

        elif asymptomatic is not None and not isinstance(asymptomatic, int):
            raise TypeError(f"asymptomatic is of unexpected type '{type(asymptomatic)}'. The expected types are int, "
                            f"float, and bool.")

        if isinstance(asymptomatic, int):
            assert asymptomatic <= num_p0s, "asymptomatic must be less than the number of num_p0s"
            state = np.ones(num_p0s) * UserStates.infected
            state[:asymptomatic] = UserStates.asymptomatic

        severity = np.random.choice(range(len(SymptomLevels) - 1), num_p0s, )
        infected = np.random.choice(range(len(self.population)), num_p0s)
        self.states[infected, 0] = state
        self.symptom_levels[infected, 0] = severity
        self.duration_infection[infected, 0] = self.duration_distribution(size=num_p0s)
        self.time_of_infection[infected, 0] = t
        outcomes = np.random.choice([UserStates.immune, UserStates.deceased],
                                    size=num_p0s,
                                    p=[1 - self.death_rate, self.death_rate])

        for i, o in zip(infected, outcomes):
            self.outcome[i] = o

    @cache_manager
    def distances(self):
        positions = self.population.position
        return distance(positions)

    def r(self):
        # total = sum(self.particles_infected.ravel() > 0)
        candidates = (self.particles_infected.ravel() > 0)
        if not any(candidates):
            return 0

        return sum(self.particles_infected.ravel()) / len(self.particles_infected[candidates])

    def r_current(self):
        active = (self.states == UserStates.asymptomatic) | (self.states == UserStates.infected)
        total = sum(self.particles_infected[active].ravel() > 0)
        vectors = sum(self.particles_infected[active & (self.particles_infected > 0)])
        if total == 0.:
            return 0.

        r = vectors / total
        return r

    def get_totals(self):
        counts = {s: (self.states.ravel() == s).sum() for s in UserStates}
        return counts

    def switch_death_rate(self):
        dr = self.__death_rate
        self.__death_rate = self.death_rate
        self.death_rate = dr

    def step(self, t):
        # Update everyone's status
        active = (self.states == UserStates.asymptomatic) | (self.states == UserStates.infected)
        candidates = active & ((self.time_of_infection * -1) + t > self.duration_infection)
        for id_ in np.where(candidates)[0]:
            self.states[id_][0] = self.outcome[id_]
            if self.states[id_] == UserStates.deceased:
                self.population[id_].motion_mask[:] = False

        distances = self.distances()
        contacts = np.argwhere(distances <= self.radius)
        contacts = contacts[contacts[:, 0] != contacts[:, 1], :]
        ids = set(contacts[:, 0])
        infected = 0
        new_infections = {}
        for id_ in ids:
            if not self.states[id_] == UserStates.susceptible:
                continue

            if hasattr(self.population[id_], "isolated") and self.population[id_].isolated:
                continue

            contact_ids = contacts[contacts[:, 0] == id_, 1]
            self.contacts[id_].update(contact_ids, t, use_last=True)
            sc_idx = np.array([c_id for c_id, e in self.contacts[id_].contacts.items()
                               if e.current >= self.exposure_time], dtype=int)

            sufficient_contact = np.zeros((len(self.population), 1), dtype=bool)
            sufficient_contact[sc_idx] = True

            infectious = (sufficient_contact &
                          ((self.states == UserStates.asymptomatic) |
                           (self.states == UserStates.infected))
                          ).ravel()
            if sum(sufficient_contact) > 0 and any(infectious):
                infected += 1
                new_infections[id_] = np.random.choice([UserStates.asymptomatic, UserStates.infected])
                if new_infections[id_] == UserStates.infected:
                    self.symptom_levels[id_] = np.random.choice(list(SymptomLevels)[2:])
                else:
                    self.symptom_levels[id_] = SymptomLevels.no_symptoms

                self.duration_infection[id_] = self.duration_distribution()
                self.time_of_infection[id_] = t
                dr = self.__death_rate
                if (self.symptom_levels[id_] == SymptomLevels.severe and self.icu_beds is not None and
                        sum(self.symptom_levels == SymptomLevels.severe) > self.icu_beds):
                    dr = self.__death_rate_icu

                self.outcome[id_] = np.random.choice([UserStates.immune, UserStates.deceased], p=[1 - dr, dr])
                self.particles_infected[infectious, 0] = self.particles_infected[infectious, 0] + (1 / sum(infectious))

        self.states[list(new_infections.keys()), 0] = list(new_infections.values())

        return self.states, self.symptom_levels, infected
