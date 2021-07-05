import random
from ._room import BaseRoom
from matplotlib.patches import Rectangle
import numpy as np
import numpy_indexed as npi

"""
    :param public: marks floor as a public section
    :type public: boolean, optional
"""


class Corridor(BaseRoom):
    def __init__(self, dims=(1, 10.5), public=False, floor_number=0, **kwargs):
        super().__init__(dims=dims, **kwargs)
        self.public = public
        self.floor_number = floor_number
        # rotate room
        if self.rotation == 90 or self.rotation == 270:
            self.dims[0], self.dims[1] = self.dims[1], self.dims[0]
        self.__room_entries = []
        self.__adjacent_rooms = []
        self.furniture_origins = None
        self.furniture_upper = None

    def _draw_world(self, ax, bbox=False, origin=(0, 0)):
        abs_origin = self.origin + origin
        if self.public:
            ax.add_patch(
                Rectangle(abs_origin, *self.dims, fill=True, linewidth=1.2, edgecolor='blue', facecolor="blue",
                          alpha=0.2))
        else:
            ax.add_patch(Rectangle(abs_origin, *self.dims, fill=False, linewidth=1.2, edgecolor='gray'))
        self.door.draw(ax, bbox, abs_origin)

    def get_room_entries(self):
        return self.__room_entries, self.__adjacent_rooms

    def set_room_entries(self, room_entry, id):
        self.__room_entries += room_entry
        self.__adjacent_rooms += id

    def enter_world(self, n, idx=None, locations=None):
        if idx is None:
            return [self.entry_point] * n
        entries, rooms = np.array(self.__room_entries), np.array(self.__adjacent_rooms)
        indices = npi.indices(rooms, locations)
        return entries[indices]

    def exit_world(self, idx):
        bool_ix = self.population.find_indexes(idx)
        self.population.motion_mask[bool_ix] = True

    def step(self, t):
        if self.public:
            return
        if not hasattr(self, "population"):
            return
        if not self.population:
            return

        no_target = np.isnan(self.population.target)
        no_target = np.array([all(i) for i in no_target])
        if no_target.any():
            living = self.__room_entries[self.__adjacent_rooms.index(id(self.population.home[0].livingroom))]
            for i in self.population.bedroom[no_target]:
                bedroom = self.__room_entries[self.__adjacent_rooms.index(id(i))]
                idx = random.randint(0, 1)
                entries = [living, bedroom]
                self.population.target[no_target] = entries[idx]
