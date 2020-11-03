from masskrug.motion.base_motion import Motion
from masskrug.pathogen import UserStates, SymptomLevels
from masskrug.worlds import World, CompositeWorld


class Ambulance(Motion):
    def __init__(self, world: CompositeWorld, population, hospital, symptom_level=SymptomLevels.strong):
        super().__init__(world, population)
        self.symptom_level = symptom_level
        self.hospital = hospital

    def step(self, t):
        active = self.population.state == UserStates.infected
        hospitalize = ((self.population.symptom_level >= self.symptom_level).ravel()
                       & active.ravel() & (self.population.location != self.hospital))
        if not any(hospitalize):
            return

        self.population.motion_mask[hospitalize.ravel()] = False

        move_to_hospital = hospitalize.ravel()
        if move_to_hospital.any():
            self.world.move_particles(move_to_hospital, self.hospital)
