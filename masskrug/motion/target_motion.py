import numpy as np

from masskrug.engine.agents import AgentList
from masskrug.motion.base_motion import Motion
from masskrug.utils import cache_manager


class MoveToTarget(Motion):
    def __init__(self, world, population: AgentList, speed=0.7):
        super().__init__(world, population)
        self.radius = 0
        n = len(population)
        self.speed = speed

        if hasattr(population, "target"):
            self.target = self.population.target
        else:
            # self.target = np.copy(self.population.position)  # copy to prevent same memory
            self.target = np.full(self.population.position.shape, np.nan)  # start with random motion
            population.add_property("target", self.target)

        if hasattr(population, "inter_target"):
            self.inter_target = self.population.inter_target
        else:
            self.inter_target = np.full(self.population.position.shape, np.nan)
            population.add_property("inter_target", self.inter_target)

        self.arrived = self.target.copy()

    def update_positions(self, t):

        if not np.all(np.isnan(self.target)):
            t = ~np.isnan(self.target) & np.isnan(self.inter_target)
            t = [all(i) for i in t]

            positions = self.population.position[t, :]
            difference = np.zeros((len(self.population), 2))
            difference[t, :] = positions - self.target[t, :]
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

        # inter target motion for walking around furniture
        if not np.all(np.isnan(self.inter_target)):
            t = ~np.isnan(self.inter_target)
            t = np.array([all(i) for i in t])

            # check if inter_target is reachable
            inter_targets = self.inter_target[t]
            dims = np.array([i.dims for i in self.population.location[t]])
            check_top = (inter_targets[:, 0] > dims[:, 0]) & ~np.isclose(inter_targets[:, 0], dims[:, 0])
            check_right = (inter_targets[:, 1] > dims[:, 1]) & ~np.isclose(inter_targets[:, 0], dims[:, 1])
            check_left = (inter_targets[:, 0] < 0) & ~np.isclose(inter_targets[:, 0], 0)
            check_bottom = (inter_targets[:, 1] < 0) & ~np.isclose(inter_targets[:, 0], 0)

            out_dim = np.zeros(t.shape, dtype=bool)
            out_dim[t] = (check_left | check_top | check_bottom | check_right)
            self.inter_target[out_dim] = np.nan

            t = t & ~out_dim

            if t.any():
                positions = self.population.position[t, :]
                difference = np.zeros((len(self.population), 2))
                difference[t, :] = positions - self.inter_target[t, :]
                distance_to_target = np.sqrt((difference ** 2).sum(axis=1))

                update = distance_to_target > 0
                step_size = (distance_to_target - self.radius) / self.speed
                step_size[step_size.ravel() > 1] = 1
                step_size *= self.speed

                x_direction = difference[update, 0] / distance_to_target[update]
                y_direction = difference[update, 1] / distance_to_target[update]
                self.positions[update, 0] = self.positions[update, 0] - x_direction * step_size[update]
                self.positions[update, 1] = self.positions[update, 1] - y_direction * step_size[update]
                arrived = np.isclose(self.positions[t], self.inter_target[t, :])
                arrived = np.array([all(i) for i in arrived])
                inter = np.copy(self.inter_target[t])
                inter[arrived] = np.nan
                self.inter_target[t] = inter

            # If motion precedes any distance computing module, we need to invalidate the cache to avoid using old
            # distances.
            cache_manager.invalidate()
