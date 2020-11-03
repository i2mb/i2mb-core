import numpy as np
from masskrug.engine.particle import ParticleList
from masskrug.pathogen import UserStates
from masskrug.pathogen.base_pathogen import Pathogen
from masskrug.utils.spatial_utils import region_ravel_multi_index


class RegionVirusDynamicExposure(Pathogen):
    """
    Pathogen model that uses an `exposure_function` to determine the level of exposure that would cause an agent to
    become infectious. The exposure_function is balanced by the `recovery_function` which reduces the level of infection
    of an agent. In addition, this class can take a function that determines the infectiousness level of a particle.

    The exposure_function has following signature:

        function(population, contacts_with_with_distance, time)

    The signature of the `recovery_function` and  the`infectiousness_function` is as follows:

        function(population, time)


    """

    def __init__(self, exposure_function, recovery_function, infectiousness_function, population: ParticleList):
        self.population = population

        self.infectiousness_function = infectiousness_function
        self.recovery_function = recovery_function
        self.exposure_function = exposure_function

        shape = (len(population), 1)
        self.exposure = np.zeros(shape)
        self.infectiousness = np.zeros(shape)

    def update_states(self, t):
        exposed = self.exposure > 1 & self.population.state == UserStates.susceptible
        self.population.state[exposed] = UserStates.incubation
        self.exposure[exposed] = 0

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
            self.particles_infected[region.population.index[region_contagions]] += infected_count

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
