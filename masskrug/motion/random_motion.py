import numpy as np

from .base_motion import Motion
from masskrug.utils import cache_manager


class RandomMotion(Motion):
    def __init__(self, world, population, gravity=None, step_size=1):
        super().__init__(world, population)
        self.gravity_field = gravity
        self.step_size = step_size
        if hasattr(self.population, "motion_mask"):
            self.motion_mask = population.motion_mask

    def update_positions(self, t):
        if hasattr(self.population, "target"):
            if np.any(np.isnan(self.population.target)) : # prevent overlay of target motion and random motion

                t = np.isnan(self.population.target)
                t = [all(i) for i in t]

                positions = self.population.position[t,:]
                num_agents = len(positions)
                direction = np.random.random((num_agents, 1)) * 2 * np.pi
                mask = np.zeros(len(self.population), dtype=bool)
                mask[t] = self.motion_mask[t].ravel()
                if not mask.any():
                    return


                direction = np.random.random((mask.sum(), 2)) * 2 - 1

                self.population.position[mask] += direction * self.step_size

                if self.gravity_field is not None:
                    positions[mask, :] += self.gravity_field

                # If motion precedes any distance computing module, we need to invalidate the cache to avoid using old
                # distances.
                cache_manager.invalidate()
