from copy import copy

import numpy as np


def vectorized(fn):
    ParticleList.particle_properties.append(fn.__name__)
    return fn


class ArrayView:
    def __init__(self, parent, p_index):
        self.__parent = parent
        self.__p_index = p_index
        self.__cv = parent[p_index]

    def __setitem__(self, key, value):
        key = self.__p_index[key]
        self.__parent[key] = value

    def __getitem__(self, item):
        return self.__cv.__getitem__(item)

    def __getattribute__(self, item):
        if item in [f"_ArrayView__{v}" for v in ["parent", "p_index", "cv"]]:
            return object.__getattribute__(self, item)

        return object.__getattribute__(self, "_ArrayView__cv").__getattribute__(item)

    def __invert__(self):
        return ~self.__cv

    def __eq__(self, other):
        return self.__cv == other


class ParticleListView:
    def __init__(self, particles, index):
        self.__source = particles
        self.__index = index
        self.__len = len(index)

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, v):
        self.__index = v
        self.__len = len(v)

    def __len__(self):
        return self.__len

    def __getitem__(self, item):
        return self.__source.getitem(self.index)[item]

    def __getattribute__(self, item):
        if item not in ParticleList.particle_properties:
            return object.__getattribute__(self, item)

        attribute = self.__source.__getattribute__(item)
        a_view = ArrayView(attribute, self.index)
        return a_view

    def find_indexes(self, idx):
        return (self.__index.reshape((-1, 1)) == idx.ravel()).any(axis=1).ravel()


class ParticleList:
    particle_properties = []
    list_properties = []

    def __init__(self, particles=0):
        if not isinstance(particles, int):
            self.index = np.array(particles)
        else:
            self.index = np.array(range(particles))

        self.__particles = np.array([Particle(id_) for id_ in self.index])

        self.__view = False

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
        return len(self.index)

    def __iter__(self):
        return iter(self.__particles)

    def getitem(self, item):
        return self.__particles[item]

    def __getitem__(self, item):
        """:return Particle item, or ParticleList view with selected particles."""
        if isinstance(item, int) or isinstance(item, np.integer):
            return self.__particles[item]

        view = ParticleListView(self, self.index[item])
        return view

    def __getattribute__(self, item):
        if item in ParticleList.particle_properties and self.__view:
            return object.__getattribute__(self, item)[self.index]

        return object.__getattribute__(self, item)

    def add_property(self, prop, values, l_property=False):
        if self.__view:
            raise RuntimeWarning("Adding properties to ParticleList views si not supported.")

        object.__setattr__(self, prop, values)

        if l_property:
            ParticleList.list_properties.append(prop)
            return

        ParticleList.particle_properties.append(prop)

        # If values is an numpy array, we want to keep a pointer.
        for p, v in zip(self.__particles, values):
            object.__setattr__(p, prop, v)


class Particle:
    def __init__(self, id_):
        self.__id = id_

    @property
    def id(self):
        return self.__id

