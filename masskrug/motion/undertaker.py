from masskrug.motion.base_motion import Motion
from masskrug.pathogen import UserStates
from masskrug.worlds import World, CompositeWorld


class Undertaker(Motion):
    def __init__(self, world: CompositeWorld, population, graveyard):
        super().__init__(world, population)
        self.graveyard = graveyard

    def step(self, t):
        deceased = self.population.state == UserStates.deceased
        if not any(deceased):
            return

        self.population.motion_mask[deceased.ravel()] = False

        move_to_graveyard = deceased.ravel() & (self.population.location != self.graveyard)
        self.world.move_agents(move_to_graveyard, self.graveyard)
