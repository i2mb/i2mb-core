from collections import Counter

import numpy as np

from i2mb.engine.agents import AgentList
from i2mb.pathogen import UserStates
from i2mb.pathogen.base_pathogen import Pathogen, SymptomLevels
from i2mb.utils import global_time
from i2mb.utils.spatial_utils import contacts_within_radius


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

    def __init__(self, exposure_function, recovery_function, infectiousness_function, population: AgentList,
                 radius=5,
                 illness_duration_distribution=None,
                 infectious_duration_pso_distribution=None,
                 incubation_duration_distribution=None,
                 symptom_distribution=None, death_rate=0.05, icu_beds=None,
                 locations_of_interest=None):
        Pathogen.__init__(self, population)

        self.icu_beds = icu_beds
        self.radius = radius ** 2
        self.incubation_duration_distribution = incubation_duration_distribution

        if infectious_duration_pso_distribution is None:
            infectious_duration_pso_distribution = illness_duration_distribution

        self.infectious_duration_pso_distribution = infectious_duration_pso_distribution
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

        self.create_disease_profile()

    def create_disease_profile(self):
        n = len(self.population)
        self.incubation_duration[:] = self.incubation_duration_distribution(n).reshape(-1, 1)
        self.illness_duration[:] = self.illness_duration_distribution(n).reshape(-1, 1)
        self.infectious_duration_pso[:] = self.infectious_duration_pso_distribution(n).reshape(-1, 1)

    def infect_particles(self, infected, t, asymptomatic=None, skip_incubation=False, symptoms_level=None):
        num_p0s = len(infected)

        asymptomatic = self.__get_asymptomatic(asymptomatic, num_p0s)
        severity = self.__get_severity_levels(asymptomatic, num_p0s, symptoms_level)

        state = UserStates.infected
        incubation_period = self.incubation_duration[infected]
        t_infection = t
        if skip_incubation:
            state = UserStates.infectious
            t_infection = t - incubation_period.ravel()

        self.__infect_particles(infected, num_p0s, severity, state, t_infection)

    def __get_severity_levels(self, asymptomatic, num_p0s, symptoms_level):
        if symptoms_level is not None:
            return np.full(num_p0s, symptoms_level, dtype=int)

        if asymptomatic is None:
            severity = np.random.choice(SymptomLevels.full_symptom_levels(), num_p0s, p=self.symptom_distribution)

        else:
            symptom_distro = np.array(self.symptom_distribution.copy())
            a_p = symptom_distro[SymptomLevels.no_symptoms]
            distribute_a_p = len(symptom_distro) - 1
            symptom_distro[1:] += a_p / distribute_a_p
            symptom_distro[SymptomLevels.no_symptoms] = 0
            severity = np.random.choice(SymptomLevels.full_symptom_levels(), num_p0s, p=symptom_distro)
            severity[:asymptomatic] = SymptomLevels.no_symptoms

        return severity

    def __get_asymptomatic(self, asymptomatic, num_p0s):
        """Returns the number of asymptomatic to insert in this call. This function returns None when asymptomatic
        agents are desired but the actual number is drawn from the specified distribution. See '__get_severity_levels(
        ...)'"""
        if asymptomatic is None:
            return None

        if type(asymptomatic) is bool:
            if asymptomatic:
                return None
            else:
                return 0

        if isinstance(asymptomatic, float):
            if 0 <= asymptomatic <= 1:
                asymptomatic = int(num_p0s * asymptomatic)
            else:
                # We understand numbers greater than one as the number of asymptomatic agents.
                asymptomatic = int(asymptomatic)

        elif type(asymptomatic) != int:
            raise RuntimeError(f"Type {type(asymptomatic)} of asymptomatic not supported")

        assert asymptomatic <= num_p0s, "asymptomatic must be less or equal than the number of num_p0s"

        return asymptomatic

    def __get_outcomes(self, num_p0s):
        return np.random.choice([UserStates.immune, UserStates.deceased],
                                size=num_p0s,
                                p=[1 - self.death_rate, self.death_rate])

    def __infect_particles(self, infected, num_p0s, severity, state, t_infection):
        self.states[infected, 0] = state
        self.symptom_levels[infected, 0] = severity
        self.time_of_infection[infected, 0] = t_infection
        self.location_contracted[infected, 0] = [type(loc).__name__.lower() for loc in
                                                 self.population.location[infected]]
        # TODO: Very ugly fix.
        at_home = self.population.at_home[infected]
        self.location_contracted[infected[at_home], 0] = "home"

        self.outcomes[infected, 0] = self.__get_outcomes(num_p0s)

    def update_states(self, t):
        self.move_susceptible_exposed()
        self.update_exposed(t)
        infected = self.move_infected_to_infectious(t)
        active = self.move_infectious_to_recovered(t)
        self.update_infectiousness_level(active, infected, t)

        # Update death rate as a function of ICU bed occupation (Critical patients)
        self.death_rate = self.__death_rate
        if ((self.symptom_levels[active] == SymptomLevels.strong).any() and self.icu_beds is not None and
                sum(self.symptom_levels[active] == SymptomLevels.strong) > self.icu_beds):
            self.death_rate = self.__death_rate_icu

    def update_infectiousness_level(self, active, infected, t):
        # Update infectiousness level
        self.infectiousness_level[active | infected] = (
            self.infectiousness_function(t - self.time_of_infection[active | infected],
                                         self.incubation_duration[active | infected],
                                         self.infectious_duration_pso[active | infected]))

    def move_infectious_to_recovered(self, t):
        # Particles that have gone through the decease.
        active = self.states == UserStates.infectious
        if active.any():
            through = (self.infectious_duration_pso +
                       self.incubation_duration +
                       self.time_of_infection) <= t
            if through.any():
                self.states[active & through] = self.outcomes[active & through]
        return active

    def move_infected_to_infectious(self, t):
        # Update particle states, infected
        infected = self.states == UserStates.infected
        if infected.any():
            infectious = (self.incubation_duration +
                          self.time_of_infection) <= t
            if infectious.any():
                self.states[infected & infectious] = UserStates.infectious
        return infected

    def update_exposed(self, t):
        # Agents that are exposed
        exposed = self.states == UserStates.exposed
        if exposed.any():
            self.time_exposed[exposed] += 1
            # Fixing 0 approximations
            self.move_exposed_to_susceptible(exposed)

            infected = (self.exposure >= .99) & exposed
            if infected.any():
                self.move_exposed_to_infected(infected, t)
                exposed[infected] = False

            self.exposure[exposed] = self.recovery_function(t, self.time_exposed[exposed],
                                                            self.exposure[exposed])

    def move_exposed_to_infected(self, infected, t):
        new_infected_ids = self.population.index[infected.ravel()]
        self.infect_particles(new_infected_ids, t, asymptomatic=True)
        self.exposure[infected] = 0

    def move_exposed_to_susceptible(self, exposed):
        recovered = self.exposure <= 1e-80
        if recovered.any():
            self.exposure[recovered & exposed] = 0
            self.time_exposed[recovered & exposed] = 0
            self.states[recovered & exposed] = UserStates.susceptible

    def move_susceptible_exposed(self):
        newly_exposed = (self.exposure > 1e-80) & (self.states == UserStates.susceptible)
        if newly_exposed.any():
            self.states[newly_exposed] = UserStates.exposed

    def step(self, t):
        # Freeze the deceased.
        deceased = self.states == UserStates.deceased
        if hasattr(self.population, "motion_mask"):
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

        return


class RegionVirusDynamicExposureBaseOnViralLoad(RegionVirusDynamicExposure):
    def __init__(self, exposure_function, recovery_function, infectiousness_function, population: AgentList,
                 radius=5,
                 clearance_duration_distribution=None,
                 illness_duration_distribution=None,
                 proliferation_duration_distribution=None,
                 max_viral_load_distribution=None,
                 symptom_onset_estimator=None,
                 symptom_distribution=None, death_rate=0.05, icu_beds=None, max_viral_load=1,
                 min_viral_load=1e-80):

        self.symptom_onset_estimator = symptom_onset_estimator
        self.proliferation_duration_distribution = proliferation_duration_distribution
        self.clearance_duration_distribution = clearance_duration_distribution
        self.max_viral_load_distribution = max_viral_load_distribution

        shape = (len(population), 1)
        self.clearance_duration = np.zeros(shape)
        self.proliferation_duration = np.zeros(shape)
        self.max_viral_load = np.zeros(shape)

        super().__init__(exposure_function, recovery_function, infectiousness_function, population, radius=radius,
                         illness_duration_distribution=illness_duration_distribution,
                         infectious_duration_pso_distribution=None,
                         incubation_duration_distribution=None,
                         symptom_distribution=symptom_distribution, death_rate=death_rate, icu_beds=icu_beds)

        # Normalize with max
        self.max_viral_load /= max_viral_load
        self.min_viral_load = min_viral_load / max_viral_load

    def create_disease_profile(self):
        n = len(self.population)
        self.clearance_duration[:] = self.clearance_duration_distribution(n).reshape(-1, 1)
        self.proliferation_duration[:] = self.proliferation_duration_distribution(n).reshape(-1, 1)
        self.proliferation_duration[:] += 1 * global_time.time_scalar
        self.max_viral_load[:] = self.max_viral_load_distribution(n).reshape(-1, 1)

        self.incubation_duration[:] = self.symptom_onset_estimator(self.proliferation_duration,
                                                                   self.clearance_duration, self.max_viral_load)
        self.infectious_duration_pso[:] = (self.clearance_duration[:] + self.proliferation_duration[:] -
                                           self.incubation_duration[:])

        if self.illness_duration_distribution is None:
            self.illness_duration[:] = self.infectious_duration_pso[:]
        else:
            self.illness_duration[:] = self.illness_duration_distribution(n).reshape(-1, 1)

    def update_infectiousness_level(self, active, infected, t):
        # Update infectiousness level
        self.infectiousness_level[active | infected] = (
            self.infectiousness_function(t - self.time_of_infection[active | infected],
                                         self.incubation_duration[active | infected],
                                         self.infectious_duration_pso[active | infected],
                                         self.max_viral_load[active | infected]))

    # def update_exposed(self, t):
    #     # Agents that are exposed
    #     exposed = self.states == UserStates.exposed
    #     if exposed.any():
    #         self.time_exposed[exposed] += 1
    #         # Fixing 0 approximations
    #         self.move_exposed_to_susceptible(exposed)
    #
    #         infected = (self.exposure >= self.min_viral_load) & exposed
    #         if infected.any():
    #             self.move_exposed_to_infected(infected, t)
    #             exposed[infected] = False
    #
    #         self.exposure[exposed] = self.recovery_function(t, self.time_exposed[exposed],
    #                                                         self.exposure[exposed])
