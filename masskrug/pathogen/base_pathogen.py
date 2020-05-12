import enum

from masskrug.engine.model import Model


class UserStates(enum.IntEnum):
    susceptible = 0
    immune = 1
    deceased = 2
    asymptomatic = 3
    infected = 4


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
