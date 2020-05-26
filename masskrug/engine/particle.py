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

        # Flag to speed up processing
        self.__updated = True
        self.__update_time = None
        self.__current_time = True

    def set_time(self, t):
        self.__current_time = t

    @property
    def updated(self):
        if self.__current_time == self.__update_time:
            return not self.__updated

        return self.__updated

    @updated.setter
    def updated(self, v):
        if self.__current_time == self.__update_time:
            # Prevent switching off the update flag if another module already changed it to True.
            self.__updated |= v
        else:
            self.__updated = v

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

        # If values is an numpy array, we want to keep a pointer.
        for p, v in zip(self.__particles, values):
            object.__setattr__(p, prop, v)


class Particle:
    def __init__(self, id_):
        self.__id = id_

    @property
    def id(self):
        return self.__id

