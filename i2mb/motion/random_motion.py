import numpy as np

from .base_motion import Motion
from i2mb.utils import cache_manager


class RandomMotion(Motion):
    def __init__(self, population, gravity=None, step_size=1.):
        super().__init__(population)
        self.gravity_field = gravity
        self.step_size = float(step_size)

    def update_positions(self, t):
        positions = self.population.position
        mask = self.motion_mask.ravel()
        if hasattr(self.population, "target"):
            # Allow movement of agents without targets
            mask = np.isnan(self.population.target).any(axis=0)

        if not mask.any():
            return

        direction = np.random.random((mask.sum(), 2)) * 2. - 1.
        positions[mask] += direction * self.step_size

        if self.gravity_field is not None:
            positions[mask, :] += self.gravity_field

        if hasattr(self.population, "gravity"):
            positions[mask, :] += self.population.gravity[mask, :]

        # If motion precedes any distance computing module, we need to invalidate the cache to avoid using old
        # distances.
        cache_manager.invalidate()

        return mask
