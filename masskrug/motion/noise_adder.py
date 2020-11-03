import numpy as np

from masskrug.motion.base_motion import Motion
from masskrug.utils import cache_manager


class NoiseAdder(Motion):
    def __init__(self, world, population, step_size=0.01):
        super().__init__(world, population)
        self.step_size = step_size

    def update_positions(self, t):
        positions = self.population.position
        num_agents = len(positions)
        direction = np.random.random((num_agents, 1)) * 2 * np.pi
        mask = self.motion_mask.ravel()
        positions[mask, :] += (np.hstack((np.cos(direction), np.sin(direction))) * self.step_size)[mask, :]

        # If motion precedes any distance computing module, we need to invalidate the cache to avoid using old
        # distances.
        cache_manager.invalidate()
