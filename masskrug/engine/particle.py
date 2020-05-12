import numpy as np


def vectorized(fn):
    ParticleList.particle_properties.append(fn.__name__)
    return fn


class ParticleList:
    particle_properties = []

    def __init__(self, particles):
        self.__len = particles
        if not isinstance(particles, int):
            self.index = np.array(particles)
            self.__len = len(self.index)
        else:
            self.index = np.array(range(self.__len))

        self.__particles = [Particle(id_) for id_ in self.index]

    def __len__(self):
        return self.__len

    def __iter__(self):
        return iter(self.__particles)

    def __getitem__(self, item):
        """:return Particle item"""
        # TODO: We need an index based getter method
        return self.__particles[item]

    def add_property(self, prop, values):
        object.__setattr__(self, prop, values)
        for p, v in zip(self.__particles, values):
            object.__setattr__(p, prop, v)


class Particle:
    def __init__(self, id_):
        self.__id = id_

    @property
    def id(self):
        return self.__id
