from i2mb.motion.base_motion import Motion
from i2mb.pathogen import UserStates, SymptomLevels
from i2mb.worlds import World, CompositeWorld


class Ambulance(Motion):
    def __init__(self, population, relocator, hospital, symptom_level=SymptomLevels.strong):
        super().__init__(population)
        self.relocator = relocator
        self.symptom_level = symptom_level
        self.hospital = hospital

    def step(self, t):
        active = self.population.state == UserStates.infectious
        hospitalize = ((self.population.symptom_level >= self.symptom_level).ravel()
                       & active.ravel() & (self.population.location != self.hospital))
        if not any(hospitalize):
            return

        self.population.motion_mask[hospitalize.ravel()] = False

        move_to_hospital = hospitalize.ravel()
        if move_to_hospital.any():
            self.relocator.move_agents(move_to_hospital, self.hospital)
