import enum

from masskrug.engine.model import Model


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
    infected = 3
    infectious = 4

    @staticmethod
    def contagious():
        return UserStates.infectious


class SymptomLevels(enum.IntEnum):
    # Really not a problem
    not_sick = -1

    # Asymptomatic patient
    no_symptoms = 0
    mild = 1
    strong = 2
    severe = 3

    @staticmethod
    def symptom_levels():
        return SymptomLevels.mild, SymptomLevels.strong, SymptomLevels.severe


class Pathogen(Model):
    def __init__(self):
        self.waves = []
        self.wave_done = True

    def update_wave_done(self, pandemic_active, t):
        if not pandemic_active and self.wave_done is False:
            self.wave_done = True
            self.waves[-1][1] = t
