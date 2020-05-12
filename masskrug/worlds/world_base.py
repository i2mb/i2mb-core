from abc import ABC, abstractmethod

import numpy as np

from masskrug.engine.model import Model


class World(Model):
    def __init__(self):
        self._dims = None
        self.landmarks = []

    @property
    def space_dims(self):
        return self._dims

    def random(self, n):
        """Generates `n` random positions in the world."""
        return np.random.random((n, 2)) * self._dims

    @abstractmethod
    def check_positions(self, *args):
        pass


class Landmark(ABC):
    def __init__(self, world):
        self.world = world
        self.world.landmarks.append(self)

    @abstractmethod
    def remove_overlap(self):
        pass
