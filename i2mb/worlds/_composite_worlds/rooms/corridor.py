import numpy as np
import numpy_indexed as npi

from ._room import BaseRoom

"""
    :param public: marks floor as a public section
    :type public: boolean, optional
"""


class Corridor(BaseRoom):
    def __init__(self, dims=(1, 10.5), public=False, floor_number=0, **kwargs):
        super().__init__(dims=dims, **kwargs)
        self.public = public
        self.floor_number = floor_number

        self.__room_entries = np.empty((0, 2), dtype=float)
        self.__adjacent_rooms = np.empty(0, dtype=int)
        self.furniture_origins = None
        self.furniture_upper = None

    def get_room_entries(self):
        return self.__room_entries, self.__adjacent_rooms

    def set_room_entries(self, room_entry, id_):
        self.__room_entries = np.append(self.__room_entries, room_entry.reshape(-1, 2), axis=0)
        self.__room_entries = self.constrain_positions(self.__room_entries)

        self.points.append(self.__room_entries[-1, :])
        self.__adjacent_rooms = np.append(self.__adjacent_rooms, [id(id_)])

    def enter_world(self, n, idx=None, arriving_from=None):
        if idx is None:
            return [self.entry_point] * n

        if len(arriving_from) == 0:
            return [self.entry_point] * n

        arriving_from = np.array([id(i) for i in arriving_from])
        outsiders = (self.__adjacent_rooms.reshape(-1, 1) != arriving_from).all(axis=0)

        if id(self.parent) in self.__adjacent_rooms:
            arriving_from[outsiders.ravel()] = id(self.parent)
        else:
            arriving_from[outsiders.ravel()] = self.__adjacent_rooms[-1]

        indices = npi.indices(self.__adjacent_rooms, arriving_from)
        return self.__room_entries[indices].reshape((-1, 2))

    def step(self, t):
        if self.public:
            return

        if not hasattr(self, "population"):
            return

        if not self.population:
            return

        if not hasattr(self.population, "target"):
            return

        no_target = np.isnan(self.population.target)
        no_target = np.array([all(i) for i in no_target])
        if no_target.any():
            living = self.__room_entries[self.__adjacent_rooms.index(id(self.population.home[0].living_room))]
            for i in self.population.bedroom[no_target]:
                bedroom = self.__room_entries[self.__adjacent_rooms.index(id(i))]
                idx = np.random.randint(0, 1)
                entries = [living, bedroom]
                self.population.target[no_target] = entries[idx]
