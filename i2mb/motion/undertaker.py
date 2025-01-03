from i2mb.motion.base_motion import Motion
from i2mb.pathogen import UserStates
from i2mb.worlds import World, CompositeWorld


class Undertaker(Motion):
    def __init__(self, population, relocator, graveyard):
        super().__init__(population)
        self.relocator = relocator
        self.graveyard = graveyard

    def step(self, t):
        deceased = self.population.state == UserStates.deceased
        if not any(deceased):
            return

        self.population.motion_mask[deceased.ravel()] = False

        move_to_graveyard = deceased.ravel() & (self.population.location != self.graveyard)
        self.relocator.move_agents(move_to_graveyard, self.graveyard)
