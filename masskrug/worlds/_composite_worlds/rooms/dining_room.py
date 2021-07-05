from ._room import BaseRoom
from masskrug.worlds.furniture.tables.dining import DiningTable
import numpy as np

"""
    :param num_seats: Number of seats, values not between 1 and 8 will be ignored
    :type num_seats: int, optional
"""

class DiningRoom(BaseRoom):
    def __init__(self, dims=(4, 3), scale=1, **kwargs):
        super().__init__(dims=dims, scale=scale, **kwargs)
        self.table_assignment = {}
        x, y = self.origin[0], self.origin[1]
        width, height = self.dims[0], self.dims[1]
        offset = [0, 0, height, width, 0]  # array to compensate offset after rotation
        rot = np.radians(self.rotation)
        table_width, table_length = (2, 1)

        table_origin = [
            np.cos(rot) * (width - table_width * scale) * 0.5 - np.sin(rot) * (
                    height - table_length * scale) * 0.5 + offset[int(self.rotation / 90) + 1],
            np.sin(rot) * (width - table_width * scale) * 0.5 + np.cos(rot) * (
                    height - table_length * scale) * 0.5 + offset[int(self.rotation / 90)]]

        self.table = DiningTable(sits=8, rotation=self.rotation, scale=scale, length=table_length, width=table_width,
                                 origin=table_origin)

        # rotate room
        if self.rotation == 90 or self.rotation == 270:
            self.dims[0], self.dims[1] = self.dims[1], self.dims[0]

        self.furniture += [self.table]

        self.furniture_origins = np.empty((len(self.furniture) - 1, 2))
        self.furniture_upper = np.empty((len(self.furniture) - 1, 2))
        self.get_furniture_grid()

    def sit_particles(self, idx):
        bool_idx = (self.population.index.reshape(-1, 1) == idx).any(axis=1)
        new_idx = np.arange(len(self.population))[bool_idx]
        seats = len(new_idx)
        self.table.occupants += seats
        self.population.target[new_idx] = self.table.get_sitting_positions()[
                                          self.table.occupants - seats:self.table.occupants]

    def exit_world(self, idx):
        bool_ix = self.population.find_indexes(idx)
        self.population.motion_mask[bool_ix] = True
        self.table.occupants -= sum(bool_ix)
