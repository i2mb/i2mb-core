import numpy as np

from masskrug.engine.particle import ParticleList
from masskrug import Model
from masskrug.utils.spatial_utils import distance
from .world_base import World, Landmark


class GravityPylons(Model, Landmark):
    def __init__(self, beacons, population=None, world: World = None, gain=2, radius=1):
        Landmark.__init__(self, world)
        self.gain = gain
        self.radius = radius
        self.__distance = None
        if isinstance(beacons, (int, float)):
            self.beacons = world.random_position(int(beacons))
        else:
            self.beacons = np.array(beacons)
            if self.beacons.dtype not in [float, int, complex]:
                raise TypeError("Beacons should be an array of x, y coordinates. Accepted types are float, int and "
                                "complex")

        self.population = population
        self.gravity = np.zeros((len(population), 2))
        population.add_property("gravity", self.gravity)

    def step(self, t):
        gravity = 0
        self.__distance = distance(self.population.position, self.beacons, magnitude=False)
        dist_x, dist_y = self.__distance
        mag_2 = dist_x ** 2 + dist_y ** 2
        self.gravity[:, 0] = (1 / (mag_2 + 100) * np.sign(dist_x) * -self.gain).sum(axis=1)
        self.gravity[:, 1] = (1 / (mag_2 + 100) * np.sign(dist_y) * -self.gain).sum(axis=1)

        # self.gravity[:, :] = np.ones(self.gravity.shape) * (1 / (mag_2 - self.radius)) * self.gain

        return self.gravity

    def remove_overlap(self):
        return

        if self.__distance is None:
            return

        dx, dy = self.__distance
        distances = np.sqrt(dy ** 2 + dy ** 2)
        mask = (distances < self.radius)

        loc = np.argwhere((distances < self.radius))

        mag = np.linalg.norm(dist, axis=1) < self.radius

        dist = dist / np.linalg.norm(dist)
        self.population.position[mag, :] += dist[mag, :] * self.radius
