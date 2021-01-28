import numpy as np

from masskrug.engine.particle import ParticleList
from masskrug.motion.base_motion import Motion
from masskrug.utils import cache_manager


class MoveToTarget(Motion):
    def __init__(self, world, population: ParticleList, speed=0.7):
        super().__init__(world, population)
        self.radius = 0
        n = len(population)
        self.speed = speed

        if hasattr(population, "target"):
            self.target = self.population.target
        else:
            # Proper initialization to nan
            population.add_property("target", self.target)

        self.arrived = self.target.copy()

    def update_positions(self, t):
        positions = self.population.position
        if (self.target == np.nan).any():
            return

        difference = positions - self.target
        distance_to_target = np.sqrt((difference ** 2).sum(axis=1))

        update = distance_to_target > 0
        step_size = (distance_to_target - self.radius) / self.speed
        step_size[step_size.ravel() > 1] = 1
        step_size *= self.speed

        x_direction = difference[update, 0] / distance_to_target[update]
        y_direction = difference[update, 1] / distance_to_target[update]
        self.positions[update, 0] = self.positions[update, 0] - x_direction * step_size[update]
        self.positions[update, 1] = self.positions[update, 1] - y_direction * step_size[update]

        # If motion precedes any distance computing module, we need to invalidate the cache to avoid using old
        # distances.
        cache_manager.invalidate()
