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

        idx_bool = self.update_positions(t)
        self.check_limits(idx_bool)

    def update_positions(self, t):
        return np.zeros(len(self.population), dtype=bool)

    def check_limits(self, idx_bool):
        locations = set(self.population.location[idx_bool].ravel())

        idx = self.population.index[idx_bool].ravel()
        for location in locations:
            mask = (self.population.location[idx_bool].ravel() == location).ravel()  # noqa
            location.check_positions(idx[mask])
