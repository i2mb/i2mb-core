import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from i2mb.engine.agents import AgentList

from .world_base import World


class SquareWorld(World):
    def __init__(self, dims, population: "AgentList"):
        super().__init__()
        self._dims = dims
        self.population = population
        w = dims[1] * 0.25
        bl = dims[0] * 1.1
        self.containment = (bl, w, bl + w, 2 * w)
        self.positions = self.random_position(len(population))
        population.add_property("position", self.positions)

    def get_containment_positions(self, len_):
        bl = self.containment[:2]
        w = self.containment[1]
        return (np.random.random((len_, 2)) * w) + bl

    def get_full_dims(self):
        return self.containment[2], self._dims[1]

    def check_positions(self, mask):
        positions = self.positions
        check_top = mask.ravel() & (positions[:, 0] > self.dims[0])
        check_right = mask.ravel() & (positions[:, 1] > self.dims[1])
        check_left = mask.ravel() & (positions[:, 0] < 0)
        check_bottom = mask.ravel() & (positions[:, 1] < 0)

        positions[check_top, 0] = self.dims[0]
        positions[check_right, 1] = self.dims[1]
        positions[check_left, 0] = 0
        positions[check_bottom, 1] = 0
        for lm in self.landmarks:
            lm.remove_overlap()
