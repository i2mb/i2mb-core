from collections import deque, Counter
from functools import partial

import numpy as np

from i2mb.interactions.contact_list import ContactList
from i2mb.engine.agents import AgentList
from i2mb.utils.spatial_utils import distance, contacts_within_radius, ravel_index_triu_nd, region_ravel_multi_index
from i2mb.utils import cache_manager
from .base_pathogen import Pathogen, SymptomLevels, UserStatesLegacy as UserStates
from i2mb.interactions.contact_matrix import ContactMatrix


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

    def __init__(self, radius, exposure_time, population: AgentList, duration_distribution=None,
                 incubation_distribution=None, asymptomatic_p=0.01, death_rate=None, icu_beds=None):

        super().__init__(population)
        self.incubation_distribution = incubation_distribution
        self.radius = radius ** 2
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

        self.contacts = []
        for p in population:
            self.contacts.append(ContactList())

    def infect_particles(self, infected, t, asymptomatic=None, skip_incubation=False, symptoms_level=None):
        num_p0s = len(infected)
        infectious_state = np.ones(num_p0s) * UserStates.infected
        if isinstance(asymptomatic, float):
            if asymptomatic <= 1:
                infectious_state = np.random.choice([UserStates.infected, UserStates.asymptomatic], num_p0s,
                                                    p=[1 - asymptomatic, asymptomatic])
            else:
                # We understand numbers greater than one as the number of asymptomatic agents.
                asymptomatic = int(asymptomatic)

        elif isinstance(asymptomatic, bool):
            if asymptomatic:
                infectious_state = np.random.choice([UserStates.infected, UserStates.asymptomatic], num_p0s,
                                                    p=[1 - self.asymptomatic_p, self.asymptomatic_p])

        elif asymptomatic is not None and not isinstance(asymptomatic, int):
            raise TypeError(f"asymptomatic is of unexpected type '{type(asymptomatic)}'. The expected types are int, "
                            f"float, and bool.")

        if type(asymptomatic) == int:
            assert asymptomatic <= num_p0s, "asymptomatic must be less or equal than the number of num_p0s"
            infectious_state[:asymptomatic] = UserStates.asymptomatic

        severity = np.random.choice(SymptomLevels.symptom_levels(), num_p0s,
                                    p=[.8, .138, .062])
        if symptoms_level is not None:
            severity[:] = symptoms_level

        state = UserStates.incubation
        incubation_period = self.incubation_distribution(size=num_p0s)
        if skip_incubation:
            state = UserStates.infected
            incubation_period = 0

        self.states[infected, 0] = state
        self.particle_type[infected, 0] = infectious_state
        self.symptom_levels[infected, 0] = severity
        self.infectious_duration_pso[infected, 0] = self.duration_distribution(size=num_p0s)
        self.incubation_duration[infected, 0] = incubation_period
        self.time_of_infection[infected, 0] = t
        self.location_contracted[infected, 0] = [type(loc).__name__.lower() for loc in
                                                 self.population.location[infected]]
        self.outcomes[infected, 0] = np.random.choice([UserStates.immune, UserStates.deceased],
                                                      size=num_p0s,
                                                      p=[1 - self.death_rate, self.death_rate])

    def update_states(self, t):
        # Update everyone's status
        # Particles that change state from incubation to infection
        active = self.states == UserStates.incubation
        infectious = (self.incubation_duration +
                      self.time_of_infection) <= t
        self.states[active & infectious] = self.particle_type[active & infectious]

        # Particles that have gone through the decease.
        active = ((self.states == UserStates.asymptomatic) |
                  (self.states == UserStates.infected))
        through = (self.infectious_duration_pso +
                   self.incubation_duration +
                   self.time_of_infection) <= t
        self.states[active & through] = self.outcomes[active & through]

        # Freeze the deceased.
        deceased = self.states == UserStates.deceased
        self.population.motion_mask[deceased] = False

        # Update the death rate
        self.death_rate = self.__death_rate
        if ((self.symptom_levels[active] == SymptomLevels.strong).any() and self.icu_beds is not None and
                sum(self.symptom_levels[active] == SymptomLevels.strong) > self.icu_beds):
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

        return sum(self.particles_infected.ravel()) / sum(candidates)

    def r_current(self):
        active = (self.states == UserStates.asymptomatic) | (self.states == UserStates.infected)
        total = sum(self.particles_infected[active].ravel() > 0)
        vectors = sum(self.particles_infected[active & (self.particles_infected > 0)])
        if total == 0.:
            return 0.

        r = vectors / total
        return r

    def switch_death_rate(self):
        dr = self.__death_rate
        self.__death_rate = self.death_rate
        self.death_rate = dr

    def step(self, t):
        self.update_states(t)
        infection_mask = (self.states == UserStates.infected) | (self.states == UserStates.asymptomatic)
        pandemic_active = (infection_mask | (self.states == UserStates.incubation)).any()
        if not pandemic_active and self.wave_done is False:
            self.wave_done = True
            self.waves[-1][1] = t

        if not infection_mask.any():
            return self.states, self.symptom_levels, self.particles_infected, 0

        distances = self.distances()
        particles_in_proximity = distances <= self.radius
        np.fill_diagonal(particles_in_proximity, False)
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
            return self.states, self.symptom_levels, self.particles_infected, 0

        self.infect_particles(new_infections_ids, t)
        infection_vectors = (new_infections[:, num_infected_contacts != 0] /
                             num_infected_contacts[num_infected_contacts != 0]).sum(axis=1)
        self.particles_infected[infection_vectors > 0, 0] = (self.particles_infected[infection_vectors > 0, 0] +
                                                             infection_vectors[infection_vectors > 0])
        # self.infect_particles(new_infections_ids, t, True)
        infected = len(new_infections_ids)
        return self.states, self.symptom_levels, self.particles_infected, infected


class RegionCoronaVirus(CoronaVirus):
    def __init__(self, radius, exposure_time, population: AgentList, **kwargs):
        super().__init__(radius, exposure_time, population, **kwargs)
        self.contact_matrix = ContactMatrix(len(population))

    def step(self, t):
        self.update_states(t)
        self.contact_matrix.reset()

        # Freeze the deceased.
        deceased = self.states == UserStates.deceased
        self.population.motion_mask[deceased] = False

        infection_mask = (self.states == UserStates.infected) | (self.states == UserStates.asymptomatic)
        pandemic_active = (infection_mask | (self.states == UserStates.incubation))

        if not pandemic_active.any() and self.wave_done is False:
            self.wave_done = True
            self.waves[-1][1] = t

        if self.wave_done or not infection_mask.any():
            return self.states, self.symptom_levels, self.particles_infected, 0

        contacts = contacts_within_radius(self.population, self.radius)
        infection_mask = np.tile(infection_mask, len(self.states))
        susceptible = np.tile((self.states == UserStates.susceptible).T, (len(self.states), 1))
        infectious_susceptible_contact = infection_mask & susceptible
        for region_contacts in contacts:
            idx_ = np.ravel_multi_index(region_contacts.T, (len(self.population), len(self.population)))

            # the contacts come in a diagonal format. Therefore, we need to compli
            exposed1 = np.take(infectious_susceptible_contact | infectious_susceptible_contact.T, idx_)
            self.contact_matrix.update_contacts(region_contacts[exposed1])

        sufficient_contact = self.contact_matrix.get_sufficient_contact(self.exposure_time)

        # Remove contacts between infected agents
        sc_between_infected = pandemic_active[sufficient_contact[:, 0], 0] & pandemic_active[
            sufficient_contact[:, 1], 0]
        sufficient_contact = sufficient_contact[~sc_between_infected]

        num_infected_contacts = sufficient_contact.shape[0]
        if num_infected_contacts == 0:
            return self.states, self.symptom_levels, self.particles_infected, 0

        contacts, counts = np.unique(sufficient_contact.ravel(), return_counts=True)
        vectors = infection_mask[contacts, 0]
        new_infected_ids = contacts[~vectors]
        self.infect_particles(new_infected_ids, t, asymptomatic=True)
        vector_ids = contacts[vectors]

        # Since a particle can be in proximity of two or more infected agents, we equally distribute blame among
        # the infected agents.
        vector_matrix = ((sufficient_contact[:, 0] == vector_ids[:, None]) |
                         (sufficient_contact[:, 1] == vector_ids[:, None]))
        new_infected_matrix = ((sufficient_contact[:, 0] == new_infected_ids[:, None]) |
                               (sufficient_contact[:, 1] == new_infected_ids[:, None]))
        infected_count = (vector_matrix.dot(new_infected_matrix.T) / new_infected_matrix.sum(axis=1)).sum(axis=1)
        self.particles_infected[contacts[vectors], 0] += infected_count

        infected = new_infected_ids.shape[0]

        # no need to return things, access should be made via modules, and not through this return values.
        # return self.states, self.symptom_levels, self.particles_infected, infected
        return


class RegionCoronaVirusExposureWindow(RegionCoronaVirus):
    def __init__(self, radius, exposure_time, population: AgentList, susceptibility_window, **kwargs):

        super().__init__(radius, exposure_time, population, **kwargs)
        self.susceptibility_window = susceptibility_window
        shape = (len(population), 1)
        self.exposures = {}
        for i in range(shape[0]):
            self.exposures[i] = deque()

        self.accumulated_time = np.zeros(shape)
        self.oldest_exposure = np.full(shape, np.nan)

    def step(self, t):
        self.update_states(t)
        self.contact_matrix.reset()

        # Freeze the deceased.
        deceased = self.states == UserStates.deceased
        self.population.motion_mask[deceased] = False

        infection_mask = (self.states == UserStates.infected) | (self.states == UserStates.asymptomatic)
        pandemic_active = (infection_mask | (self.states == UserStates.incubation))

        if not pandemic_active.any() and self.wave_done is False:
            self.wave_done = True
            self.waves[-1][1] = t

        if self.wave_done or not infection_mask.any():
            return

        susceptible = (self.states == UserStates.susceptible)
        contacts = contacts_within_radius(self.population, self.radius)
        new_exposed = np.zeros_like(self.population.index, dtype=bool)
        for region_contacts, region in zip(contacts, [r for r in self.population.regions if len(r.population) > 1]):
            idx_ = np.unique(region_contacts.ravel())
            region_infectious = infection_mask[region.population.index]
            region_susceptibles = susceptible[region.population.index]

            n = len(region.population)
            idx_contact = region_ravel_multi_index(region_contacts.T, region.population.index)
            contact_matrix = region_infectious * region_susceptibles.T
            contact_matrix |= contact_matrix.T
            region_new_exposed = np.take(contact_matrix, idx_contact)
            contact_matrix[:] = False
            idx_triangle = region_ravel_multi_index(region_contacts[region_new_exposed].T, region.population.index)
            np.put(contact_matrix, idx_triangle, True)
            contact_matrix |= contact_matrix.T
            region_new_exposed = contact_matrix.any(axis=0) & region_susceptibles.ravel()
            if not region_new_exposed.any():
                continue

            region_contagions = contact_matrix.any(axis=1) & region_infectious.ravel()
            new_exposed[region.population.index] = region_new_exposed

            # Distribute blame per time, and per particle.
            time_contribution = 1 / self.exposure_time
            vector_matrix = ((region_contacts[:, 0] == region.population.index[region_contagions, None]) |
                             (region_contacts[:, 1] == region.population.index[region_contagions, None]))
            new_infected_matrix = ((region_contacts[:, 0] == region.population.index[region_new_exposed, None]) |
                                   (region_contacts[:, 1] == region.population.index[region_new_exposed, None]))
            infected_count = (vector_matrix.dot(new_infected_matrix.T) / new_infected_matrix.sum(axis=1)).sum(axis=1)
            infected_count *= time_contribution
            self.particles_infected[region.population.index[region_contagions]] += infected_count.reshape(-1, 1)

        new_exposed_idx = self.population.index[new_exposed]
        for ix in new_exposed_idx:
            exposure_queue = self.exposures[ix]
            exposure_queue.append(t)
            while exposure_queue[0] < (t - self.susceptibility_window):
                exposure_queue.popleft()

            self.accumulated_time[ix] = len(exposure_queue)
            self.oldest_exposure[ix] = exposure_queue[0]

        for ix in self.population.index[self.oldest_exposure.ravel() < (t - self.susceptibility_window)]:
            exposure_queue = self.exposures[ix]
            while len(exposure_queue) > 0 and exposure_queue[0] < (t - self.susceptibility_window):
                exposure_queue.popleft()

            self.accumulated_time[ix] = len(exposure_queue)
            if len(exposure_queue) > 0:
                self.oldest_exposure[ix] = exposure_queue[0]
            else:
                self.oldest_exposure[ix] = np.nan

        if not new_exposed.any():
            return

        new_infected = (self.accumulated_time >= self.exposure_time).ravel() & new_exposed
        new_infected_ids = self.population.index[new_infected.ravel()]
        self.infect_particles(new_infected_ids, t, asymptomatic=True)

        # no need to return things, access should be made via modules, and not through this return values.
        # return self.states, self.symptom_levels, self.particles_infected, infected
        return
