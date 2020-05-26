import enum

from masskrug.engine.model import Model


class UserStates(enum.IntEnum):
    susceptible = 0
    immune = 1
    deceased = 2
    incubation = 3
    asymptomatic = 4
    infected = 5

    @staticmethod
    def contagious():
        return UserStates.asymptomatic, UserStates.infected


class SymptomLevels(enum.IntEnum):
    # Really not a problem
    not_sick = -1

    # Asymptomatic patient
    no_symptoms = 0
    mild = 1
    strong = 2
    severe = 3


class Pathogen(Model):
    pass
