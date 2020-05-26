from functools import partial

import numpy as np

from masskrug.interactions.contact_list import ContactList
from masskrug.engine.particle import ParticleList
from masskrug.utils.spatial_utils import distance
from masskrug.utils import cache_manager
from .base_pathogen import Pathogen, SymptomLevels, UserStates


class CoronaVirus(Pathogen):
    """
    :param radius:
    :param exposure_time:
    :param incubation_distribution: Time between exposure and symptom onset, during this time the individual has
      the virus, but is not infectious.
    :type incubation_distribution: int, distribution
    :param population:
    :param asymptomatic_p:
    :param death_rate:
    :param icu_beds:
    """

    def __init__(self, radius, exposure_time, population: ParticleList,
                 duration_distribution=None,
                 incubation_distribution=None,
                 asymptomatic_p=0.01,
                 death_rate=None,
                 icu_beds=None):

        self.incubation_distribution = incubation_distribution
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
        self.incubation_period = np.zeros(shape)
        self.time_of_infection = np.zeros(shape)
        self.particles_infected = np.zeros(shape)
        self.outcomes = np.zeros(shape)
        self.particle_type = np.zeros(shape)
        self.contacts = []
        for p in population:
            self.contacts.append(ContactList())

        population.add_property("state", self.states)
        population.add_property("symptom_level", self.symptom_levels)
        population.add_property("duration_infection", self.duration_infection)
        population.add_property("incubation_period", self.incubation_period)
        population.add_property("time_of_infection", self.time_of_infection)
        population.add_property("particles_infected", self.particles_infected)
        population.add_property("particle_type", self.particle_type)
        population.add_property("outcome", self.outcomes)

    def introduce_pathogen(self, num_p0s, t, asymptomatic=None):
        susceptible = self.population.state == UserStates.susceptible
        num_p0s = len(susceptible) >= num_p0s and num_p0s or len(susceptible)
        ids = np.random.choice(range(len(susceptible)), num_p0s, replace=False)
        self.infect_particles(ids, t, asymptomatic)

    def infect_particles(self, infected, t, asymptomatic=None):
        num_p0s = len(infected)
        infectious_state = np.ones(num_p0s) * UserStates.infected
        if isinstance(asymptomatic, float):
            if asymptomatic <= 1:
                infectious_state = np.random.choice([UserStates.infected, UserStates.asymptomatic], num_p0s,
                                                    p=[1 - asymptomatic, asymptomatic])
            else:
                # We understand numbers greater than one as the number of asymptomatic particles.
                asymptomatic = int(asymptomatic)

        elif isinstance(asymptomatic, bool):
            if asymptomatic:
                infectious_state = np.random.choice([UserStates.infected, UserStates.asymptomatic], num_p0s,
                                                    p=[1 - self.asymptomatic_p, self.asymptomatic_p])

        elif asymptomatic is not None and not isinstance(asymptomatic, int):
            raise TypeError(f"asymptomatic is of unexpected type '{type(asymptomatic)}'. The expected types are int, "
                            f"float, and bool.")

        if isinstance(asymptomatic, int):
            assert asymptomatic <= num_p0s, "asymptomatic must be less than the number of num_p0s"
            infectious_state[:asymptomatic] = UserStates.asymptomatic

        severity = np.random.choice(range(len(SymptomLevels) - 1), num_p0s, )
        self.states[infected, 0] = np.ones(num_p0s) * UserStates.incubation
        self.particle_type[infected, 0] = infectious_state
        self.symptom_levels[infected, 0] = severity
        self.duration_infection[infected, 0] = self.duration_distribution(size=num_p0s)
        self.incubation_period[infected, 0] = self.incubation_distribution(size=num_p0s)
        self.time_of_infection[infected, 0] = t
        self.outcomes[infected, 0] = np.random.choice([UserStates.immune, UserStates.deceased],
                                                      size=num_p0s,
                                                      p=[1 - self.death_rate, self.death_rate])

    def update_states(self, t):
        # Update everyone's status
        # Particles that change state from incubation to infection
        active = self.states == UserStates.incubation
        infectious = (self.incubation_period +
                      self.time_of_infection) <= t
        self.states[active & infectious] = self.particle_type[active & infectious]

        # Particles that have gone through the decease.
        active = ((self.states == UserStates.asymptomatic) |
                  (self.states == UserStates.infected))
        through = (self.duration_infection +
                   self.incubation_period +
                   self.time_of_infection) <= t
        self.states[active & through] = self.outcomes[active & through]

        # Freeze the deceased.
        deceased = self.states == UserStates.deceased
        self.population.motion_mask[deceased] = False

        # Update the death rate
        self.death_rate = self.__death_rate
        if ((self.symptom_levels[active] == SymptomLevels.severe).any() and self.icu_beds is not None and
                sum(self.symptom_levels[active] == SymptomLevels.severe) > self.icu_beds):
            self.death_rate = self.__death_rate_icu

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
        self.update_states(t)
        infection_mask = (self.states == UserStates.infected) | (self.states == UserStates.asymptomatic)
        if not infection_mask.any():
            return self.states, self.symptom_levels, 0

        distances = self.distances()
        particles_in_proximity = distances <= self.radius
        np.fill_diagonal(particles_in_proximity, False)
        infection_mask = (self.states == UserStates.infected) | (self.states == UserStates.asymptomatic)
        infection_mask = np.tile(infection_mask, len(self.states))
        susceptible = np.tile((self.states == UserStates.susceptible).T, (len(self.states), 1))
        exposed = particles_in_proximity & infection_mask & susceptible
        if hasattr(self.population, "isolated"):
            ids = np.argwhere((exposed.sum(axis=0) > 0) & ~self.population.isolated[:, 0]).ravel()
        else:
            ids = np.argwhere(exposed.sum(axis=0) > 0).ravel()

        sufficient_contact = np.zeros((len(self.population), len(self.population)), dtype=bool)
        contacts = np.argwhere(exposed)
        for id_ in ids:
            contact_ids = contacts[contacts[:, 1] == id_, 0]
            self.contacts[id_].update(contact_ids, t, use_last=True)
            sc_idx = np.array([c_id for c_id, e in self.contacts[id_].contacts.items()
                               if e.current >= self.exposure_time], dtype=int)

            sufficient_contact[sc_idx, id_] = True

        new_infections = exposed & sufficient_contact
        num_infected_contacts = new_infections.sum(axis=0)
        new_infections_ids = np.argwhere(num_infected_contacts).ravel()
        if len(new_infections_ids) == 0:
            return self.states, self.symptom_levels, 0

        self.infect_particles(new_infections_ids, t)
        infection_vectors = (new_infections[:, num_infected_contacts != 0] /
                             num_infected_contacts[num_infected_contacts != 0]).sum(axis=1)
        self.particles_infected[infection_vectors > 0, 0] = (self.particles_infected[infection_vectors > 0, 0] +
                                                             infection_vectors[infection_vectors > 0])
        self.infect_particles(new_infections_ids, t, True)
        infected = len(new_infections_ids)
        return self.states, self.symptom_levels, infected
