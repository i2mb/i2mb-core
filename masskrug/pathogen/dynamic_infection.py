from collections import Counter

import numpy as np
from masskrug.engine.particle import ParticleList
from masskrug.pathogen import UserStates
from masskrug.pathogen.base_pathogen import Pathogen, SymptomLevels
from masskrug.utils.spatial_utils import region_ravel_multi_index, contacts_within_radius


class RegionVirusDynamicExposure(Pathogen):
    """
    Pathogen model that uses an `exposure_function` to determine the level of exposure that would cause an agent to
    become infectious. The exposure_function is balanced by the `recovery_function` which reduces the level of infection
    of an agent. In addition, this class can take a function that determines the infectiousness level of a particle.

    The exposure_function has following signature:

        function(population, contacts_with_distance, time)

    The signature of the `recovery_function` and  the`infectiousness_function` is as follows:

        function(population, time)
    """

    def __init__(self, exposure_function, recovery_function, infectiousness_function, population: ParticleList,
                 radius=5,
                 illness_duration_distribution=None,
                 infectiousness_duration_pso=None,
                 incubation_duration_distribution=None,
                 symptom_distribution=None, death_rate=0.05, icu_beds=None):
        Pathogen.__init__(self, population)

        self.icu_beds = icu_beds
        self.radius = radius ** 2
        self.incubation_duration_distribution = incubation_duration_distribution
        self.infectiousness_duration_pso = infectiousness_duration_pso
        self.illness_duration_distribution = illness_duration_distribution
        if symptom_distribution is None:
            symptom_distribution = [0.4, 0.4, .138, .062]

        self.symptom_distribution = symptom_distribution
        self.death_rate = death_rate
        self.population = population

        self.infectiousness_function = infectiousness_function
        self.recovery_function = recovery_function
        self.exposure_function = exposure_function

        try:
            self.__death_rate, self.__death_rate_icu = death_rate
        except TypeError:
            self.__death_rate, self.__death_rate_icu = death_rate, death_rate

        if self.__death_rate is None:
            self.__death_rate = .01

        if self.__death_rate_icu is None:
            self.__death_rate_icu = .5

        self.death_rate = self.__death_rate

        shape = (len(population), 1)
        self.infectiousness_level = np.zeros(shape)
        self.exposure = np.zeros(shape)
        self.time_exposed = np.zeros(shape)

        # Diagnostics and information
        self.infection_map = {}
        self.infected_by = {}
        self.contact_map = Counter()

    def infect_particles(self, infected, t, asymptomatic=None, skip_incubation=False, symptoms_level=None):
        num_p0s = len(infected)
        symptom_distro = self.symptom_distribution.copy()
        if asymptomatic is None:
            asymptomatic = True

        if symptoms_level is not None:
            skip_incubation = True

        if isinstance(asymptomatic, float):
            if 0 <= asymptomatic <= 1:
                symptom_distro[SymptomLevels.no_symptoms] = asymptomatic
                severity = np.random.choice(SymptomLevels.full_symptom_levels(), num_p0s,
                                            p=symptom_distro)
            else:
                # We understand numbers greater than one as the number of asymptomatic particles.
                asymptomatic = int(asymptomatic)

        elif type(asymptomatic) == bool:
            if not asymptomatic:
                symptom_distro[SymptomLevels.no_symptoms] = 0

            severity = np.random.choice(SymptomLevels.full_symptom_levels(), num_p0s, p=symptom_distro)

        elif type(asymptomatic) != int:
            raise RuntimeError(f"Type {type(asymptomatic)} of asymptomatic not supported")

        if type(asymptomatic) == int:
            assert asymptomatic <= num_p0s, "asymptomatic must be less or equal than the number of num_p0s"
            a_p = symptom_distro[SymptomLevels.no_symptoms]
            symptom_distro[SymptomLevels.mild] += a_p
            symptom_distro[SymptomLevels.no_symptoms] = 0
            symptom_distro = [symptom_distro[s] for s in SymptomLevels.symptom_levels()]
            severity = np.random.choice(SymptomLevels.full_symptom_levels(), num_p0s,
                                        p=symptom_distro)
            severity[:asymptomatic] = SymptomLevels.no_symptoms

        if symptoms_level is not None:
            severity[:] = symptoms_level

        state = UserStates.infected
        incubation_period = self.incubation_duration_distribution(size=num_p0s)
        t_infection = t
        if skip_incubation:
            state = UserStates.infectious
            t_infection = t - incubation_period

        self.states[infected, 0] = state
        self.symptom_levels[infected, 0] = severity
        self.infectious_duration_pso[infected, 0] = self.illness_duration_distribution(size=num_p0s)
        self.incubation_duration[infected, 0] = incubation_period
        self.time_of_infection[infected, 0] = t_infection
        self.location_contracted[infected, 0] = [type(loc).__name__.lower() for loc in
                                                 self.population.location[infected]]
        self.outcomes[infected, 0] = np.random.choice([UserStates.immune, UserStates.deceased],
                                                      size=num_p0s,
                                                      p=[1 - self.death_rate, self.death_rate])

    def update_states(self, t):
        newly_exposed = (self.exposure > 1e-80) & (self.states == UserStates.susceptible)
        if newly_exposed.any():
            self.states[newly_exposed] = UserStates.exposed

        # Agents that are exposed
        exposed = self.states == UserStates.exposed
        if exposed.any():
            self.time_exposed[exposed] += 1
            # Fixing 0 approximations
            recovered = self.exposure <= 1e-80
            if recovered.any():
                self.exposure[recovered & exposed] = 0
                self.time_exposed[recovered & exposed] = 0
                self.states[recovered & exposed] = UserStates.susceptible

            infected = (self.exposure >= .99) & exposed
            if infected.any():
                new_infected_ids = self.population.index[infected.ravel()]
                self.infect_particles(new_infected_ids, t, asymptomatic=True)
                self.exposure[infected] = 0
                exposed[infected] = False

            self.exposure[exposed] = self.recovery_function(t, self.time_exposed[exposed],
                                                            self.exposure[exposed])

        # Update particle states, infected
        infected = self.states == UserStates.infected
        if infected.any():
            infectious = (self.incubation_duration +
                          self.time_of_infection) <= t
            if infectious.any():
                self.states[infected & infectious] = UserStates.infectious

        # Particles that have gone through the decease.
        active = self.states == UserStates.infectious
        if active.any():
            through = (self.infectious_duration_pso +
                       self.incubation_duration +
                       self.time_of_infection) <= t
            if through.any():
                self.states[active & through] = self.outcomes[active & through]

        # Update infectiousness level
        self.infectiousness_level[active | infected] = self.infectiousness_function(t -
                                                                                    self.time_of_infection[
                                                                                        active | infected],
                                                                                    self.incubation_duration[
                                                                                        active | infected],
                                                                                    self.infectious_duration_pso[
                                                                                        active | infected]
                                                                                    )

        # Update death rate as a function of ICU bed occupation (Critical patients)
        self.death_rate = self.__death_rate
        if ((self.symptom_levels[active] == SymptomLevels.strong).any() and self.icu_beds is not None and
                sum(self.symptom_levels[active] == SymptomLevels.strong) > self.icu_beds):
            self.death_rate = self.__death_rate_icu

    def step(self, t):
        # Freeze the deceased.
        deceased = self.states == UserStates.deceased
        self.population.motion_mask[deceased] = False

        self.update_states(t)

        infection_mask = (self.states == UserStates.infectious) | (self.states == UserStates.infected)
        pandemic_active = infection_mask

        if not pandemic_active.any() and self.wave_done is False:
            self.wave_done = True
            self.waves[-1][1] = t

        if self.wave_done or not infection_mask.any():
            return

        contacts = contacts_within_radius(self.population, self.radius, return_distance=True)
        for region_contacts, region in zip(contacts, [r for r in self.population.regions if len(r.population) > 1]):
            region_contacts, distances = region_contacts
            self.contact_map.update([tuple(sorted(k)) for k in region_contacts])
            region_infectious = infection_mask[region.population.index].ravel()
            if not region_infectious.any():
                continue

            region_susceptible = ((self.states[region.population.index] == UserStates.exposed) |
                                  (self.states[region.population.index] == UserStates.susceptible)).ravel()

            if not region_susceptible.any():
                continue

            region_vector_contacts = np.logical_xor(
                ((region_contacts[:, 0].reshape(-1, 1) == region.population.index[region_infectious].ravel()).any(
                    axis=1) &
                 (region_contacts[:, 1].reshape(-1, 1) == region.population.index[region_susceptible].ravel()).any(
                     axis=1)),
                ((region_contacts[:, 1].reshape(-1, 1) == region.population.index[region_infectious].ravel()).any(
                    axis=1) &
                 (region_contacts[:, 0].reshape(-1, 1) == region.population.index[region_susceptible].ravel()).any(
                     axis=1)))

            if not region_vector_contacts.any():
                continue

            exposure = self.exposure_function(t, region_vector_contacts,
                                              region_contacts, distances,
                                              region.population.index,
                                              self.infectiousness_level)

            self.exposure[region.population.index] += exposure.reshape(-1, 1)
            for x, y in region_contacts[region_vector_contacts]:
                if self.states[x] == UserStates.infectious:
                    if self.exposure[y] >= .99 and y not in self.infected_by:
                        self.infection_map.setdefault(x, {}).setdefault(y, []).append(t)
                        self.infected_by[y] = x

                else:
                    if self.exposure[x] >= .99 and x not in self.infected_by:
                        self.infection_map.setdefault(y, {}).setdefault(x, []).append(t)
                        self.infected_by[x] = y

            # Distribute blame per time, and per particle.
            # contribution = exposure
            # vector_matrix = (
            #         ((region_contacts[:, 0].reshape(-1, 1) == region.population.index[region_infectious]) |
            #          (region_contacts[:, 1].reshape(-1, 1) == region.population.index[region_infectious])) &
            #         region_vector_contacts.reshape(-1, 1))
            # new_infected_matrix = (
            #         ((region_contacts[:, 0].reshape(-1, 1) == region.population.index[region_susceptible]) |
            #          (region_contacts[:, 1].reshape(-1, 1) == region.population.index[region_susceptible])) &
            #         region_vector_contacts.reshape(-1, 1))
            #
            # infected_count = (vector_matrix.T.dot(new_infected_matrix) / new_infected_matrix.sum(axis=0)).sum(axis=1)
            # infected_count *= contribution[region_infectious]
            #
            # self.particles_infected[region.population.index[region_infectious]] += infected_count.reshape(-1, 1)

        return
