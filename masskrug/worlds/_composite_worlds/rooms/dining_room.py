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

        table_dims = np.array([2, 1])
        table_width, table_length = table_dims
        rotation = 0
        dims = self.dims
        if self.width < self.height:
            rotation = 90
            dims = self.dims[::-1]

        self.table = DiningTable(sits=8, rotation=rotation, scale=scale, height=table_length, width=table_width,
                                 origin=[dims / 2 - table_dims / 2])

        self.add_furniture([self.table])
        # self.get_furniture_grid()

    def sit_particles(self, idx):
        bool_idx = (self.population.index.reshape(-1, 1) == idx).any(axis=1)
        new_idx = np.arange(len(self.population))[bool_idx]
        seats = len(new_idx)
        self.table.occupants += seats
        self.population.target[new_idx] = self.table.get_sitting_positions()[
                                          self.table.occupants - seats:self.table.occupants]

    def exit_world(self, idx):
        super().exit_world(idx)
        bool_ix = self.population.find_indexes(idx)
        self.table.occupants -= sum(bool_ix)
