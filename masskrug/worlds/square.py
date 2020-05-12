import numpy as np

from masskrug.engine.particle import ParticleList
from .world_base import World


class SquareWorld(World):
    def __init__(self, dims, population: ParticleList):
        super().__init__()
        self._dims = dims
        self.population = population
        w = dims[1] * 0.25
        bl = dims[0] * 1.01
        self.containment = (bl, w, bl + w, 2 * w)
        population.add_property("position", self.random_location(len(population)))

    def random_location(self, len_):
        return np.random.random((len_, 2)) * self.space_dims

    def get_containment_positions(self, len_):
        bl = self.containment[:2]
        w = self.containment[1]
        return (np.random.random((len_, 2)) * w) + bl

    def get_full_dims(self):
        return self.containment[2], self._dims[1]

    def check_positions(self, positions, mask, t):
        check_top = mask.ravel() & (positions[:, 0] > self.space_dims[0])
        check_right = mask.ravel() & (positions[:, 1] > self.space_dims[1])
        check_left = mask.ravel() & (positions[:, 0] < 0)
        check_bottom = mask.ravel() & (positions[:, 1] < 0)

        positions[check_top, 0] = self.space_dims[0]
        positions[check_right, 1] = self.space_dims[1]
        positions[check_left, 0] = 0
        positions[check_bottom, 1] = 0
        for lm in self.landmarks:
            lm.remove_overlap()
