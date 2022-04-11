import enum
import numpy as np

from i2mb.engine.model import Model


class UserStatesLegacy(enum.IntEnum):
    susceptible = 0
    immune = 1
    deceased = 2
    incubation = 3
    asymptomatic = 4
    infected = 5

    @staticmethod
    def contagious():
        return UserStates.asymptomatic, UserStates.infected


class UserStates(enum.IntEnum):
    susceptible = 0
    immune = 1
    deceased = 2
    exposed = 3
    infected = 4
    infectious = 5

    @staticmethod
    def contagious():
        return UserStates.infectious


class SymptomLevels(enum.IntEnum):
    # Really not a problem
    not_sick = -1

    # Asymptomatic patient
    no_symptoms = 0
    recovering = 1
    mild = 2
    strong = 3
    critical = 4

    @staticmethod
    def symptom_levels():
        return SymptomLevels.mild, SymptomLevels.strong, SymptomLevels.critical

    @staticmethod
    def full_symptom_levels():
        return SymptomLevels.no_symptoms, SymptomLevels.mild, SymptomLevels.strong, SymptomLevels.critical


class Pathogen(Model):
    def __init__(self, population):
        self.population = population
        self.waves = []
        self.wave_done = True

        shape = (len(population), 1)
        self.states = self.population.state
        self.symptom_levels = self.population.symptom_level
        self.infectious_duration_pso = np.zeros(shape)
        self.incubation_duration = np.zeros(shape)
        self.illness_duration = np.zeros(shape)
        self.time_of_infection = np.zeros(shape)
        self.particles_infected = np.zeros(shape)
        self.outcomes = np.zeros(shape)
        self.particle_type = np.zeros(shape)
        self.location_contracted = np.zeros(shape, dtype=object)

        population.add_property("infectious_duration_pso", self.infectious_duration_pso)
        population.add_property("incubation_duration", self.incubation_duration)
        population.add_property("illness_duration", self.illness_duration)
        population.add_property("time_of_infection", self.time_of_infection)
        population.add_property("particles_infected", self.particles_infected)
        population.add_property("particle_type", self.particle_type)
        population.add_property("outcome", self.outcomes)
        population.add_property("location_contracted", self.location_contracted)

    def update_wave_done(self, pandemic_active, t):
        if not pandemic_active and self.wave_done is False:
            self.wave_done = True
            self.waves[-1][1] = t

    def infect_particles(self, infected, t, asymptomatic=None, skip_incubation=False, symptoms_level=None):
        raise RuntimeError("Calling Pathogen infect_particle(), please implement in subclass.")

    def get_totals(self):
        counts = {s: (self.states.ravel() == s).sum() for s in UserStates}
        return counts

    def get_totals_per_symptom_level(self):
        counts = {s: (self.symptom_levels.ravel() == s).sum() for s in SymptomLevels}
        return counts

    def introduce_pathogen(self, num_p0s, t, asymptomatic=None, symptoms_level=None, skip_incubation=True):
        susceptible = self.population.state == UserStates.susceptible
        num_p0s = len(susceptible) >= num_p0s and num_p0s or len(susceptible)
        ids = np.random.choice(range(len(susceptible)), num_p0s, replace=False)
        self.start_wave(t)

        self.infect_particles(ids, t, asymptomatic, skip_incubation=skip_incubation, symptoms_level=symptoms_level)

    def start_wave(self, t):
        if self.wave_done:
            self.wave_done = False
            self.waves.append([t, None])
