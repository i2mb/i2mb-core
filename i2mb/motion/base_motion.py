import numpy as np
from i2mb.engine.model import Model


class Motion(Model):
    def __init__(self, world, population):
        self.world = world
        self.population = population
        self.positions = population.position
        if not hasattr(population, "motion_mask"):
            self.motion_mask = np.ones((len(population), 1), dtype=bool)
            population.add_property("motion_mask", self.motion_mask)

    def step(self, t):
        if not self.population.updated:
            return self.positions

        self.update_positions(t)
        self.check_limits(t)
        return self.positions

    def update_positions(self, t):
        pass

    def check_limits(self, t):
        self.world.check_positions(self.motion_mask)