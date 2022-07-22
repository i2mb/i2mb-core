import numpy as np
from i2mb.engine.model import Model


class Motion(Model):
    def __init__(self, population):
        super().__init__()
        self.population = population
        if not hasattr(population, "motion_mask"):
            self.motion_mask = np.ones((len(population), 1), dtype=bool)
            population.add_property("motion_mask", self.motion_mask)
        else:
            self.motion_mask = self.population.motion_mask

    def step(self, t):
        if not self.population.updated:
            return self.population.positions

        self.update_positions(t)
        self.check_limits(t)
        return self.positions

    def update_positions(self, t):
        pass

    def check_limits(self, t):
        self.world.check_positions(self.motion_mask)
