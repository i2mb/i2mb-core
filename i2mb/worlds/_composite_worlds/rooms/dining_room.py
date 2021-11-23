import numpy as np

import i2mb.activities.activity_descriptors
from i2mb.worlds.furniture.tables.dining import DiningTable
from ._room import BaseRoom

"""
    :param num_seats: Number of seats, values not between 1 and 8 will be ignored
    :type num_seats: int, optional
"""


class DiningRoom(BaseRoom):
    def __init__(self, dims=(4, 3), scale=1, **kwargs):
        super().__init__(dims=dims, scale=scale, **kwargs)

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

        self.local_activities.extend([
            i2mb.activities.activity_descriptors.Eat(location=self, device=self.table, duration=9, blocks_for=1)])

        # Manage sitting positions
        self.table_assignment = np.ones(len(self.table.get_sitting_positions())) *- 1

    def sit_agents(self, idx):
        required_seats = len(idx)
        available_seats = (self.table_assignment.reshape(-1,1) == -1).any(axis=1)
        bool_idx = (self.population.index.reshape(-1, 1) == idx).any(axis=1)
        if available_seats.sum() > 0:
            if required_seats > available_seats.sum():
                required_seats = available_seats.sum()

            choose_seats = np.where(available_seats)[0][:required_seats]
            self.table_assignment[choose_seats] = idx[:required_seats]

            choose_idx = np.where(bool_idx)[0][:required_seats]
            self.population.position[choose_idx] = self.table.get_sitting_positions()[choose_seats]

        if required_seats < len(idx):
            choose_idx = np.where(~bool_idx)[0][:required_seats]
            self.population.position[choose_idx] = np.random.random((len(idx) - required_seats, 2)) * self.dims

    def start_activity(self, idx, descriptor_ids):
        self.sit_agents(idx)

    def stop_activity(self, idx, descriptor_ids):
        self.raise_agents(idx)

    def exit_world(self, idx, global_population):
        super().exit_world(idx, None)
        bool_ix = self.population.find_indexes(idx)
        self.table.occupants -= sum(bool_ix)

    def raise_agents(self, idx):
        assigned_seats = (self.table_assignment.reshape(-1, 1) == idx).any(axis=1)
        self.table_assignment[assigned_seats] = -1
        bool_idx = self.population.find_indexes(idx)
        self.population.position[bool_idx] = self.dims / 2
