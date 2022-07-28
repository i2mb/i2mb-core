import numpy as np

from i2mb.activities.activity_descriptors import Work
from i2mb.utils import global_time, time
from i2mb.worlds import BaseRoom
from i2mb.worlds.furniture.tables.dining import DiningTable


class Office(BaseRoom):
    def __init__(self, dims=(10, 10), num_tables=10, seats_table=6, tables_per_row=3,
                 corridor_width=0.7, **kwargs):

        super().__init__(dims=dims, **kwargs)

        self.corridor_width = corridor_width
        self.tables_per_row = tables_per_row
        self.num_tables = num_tables
        self.seats_table = seats_table
        self.sits = 2 * num_tables

        # self.seats = np.zeros((num_tables * seats_table, 2))
        table_dims = np.array([2, 1])
        table_width, table_length = table_dims
        rotation = 0
        if self.width < self.height:
            rotation = 90

        self.tables = [DiningTable(sits=seats_table, rotation=rotation,
                                   height=table_length, width=table_width) for _ in range(num_tables)]

        self.arrange_tables()
        self.add_furniture(self.tables)
        activities = [Work(location=self, duration=global_time.make_time(hour=8), blocks_for=1)]
        self.local_activities.extend(activities)

        # Stand alone building
        self.available_activities.extend(activities)
        self.default_activity = activities[0]

        # Seat management
        self.available_seats = self.get_available_seats(num_tables * 2)
        self.seat_assignment = np.ones(num_tables * 2, dtype=int) * -1

    def arrange_tables(self):
        row = col = 0
        t_width = t_length = 0
        for table in self.tables:
            table.origin = [self.corridor_width + col * (t_width + self.corridor_width),
                            self.corridor_width + row * (t_length + self.corridor_width)]
            t_width, t_length = table.dims

            col += 1
            if col == self.tables_per_row:
                col = 0
                row += 1

    def get_available_seats(self, num_seats):
        # Get only two seats per table
        seats = np.vstack([t.get_sitting_positions()[2:4, :] for t in self.tables])
        return seats

    def raise_agents(self, idx):
        assigned_seats = (self.seat_assignment.reshape(-1, 1) == idx).any(axis=1)
        self.seat_assignment[assigned_seats] = -1
        bool_idx = self.population.find_indexes(idx)
        self.population.position[bool_idx] = self.dims / 2

    def start_activity(self, idx, activity_id):
        bool_idx = self.population.find_indexes(idx)
        self.population.motion_mask[bool_idx] = False
        self.sit_agents(idx)

    def stop_activity(self, idx, activity_id):
        bool_idx = self.population.find_indexes(idx)
        self.population.motion_mask[bool_idx] = True
        self.raise_agents(idx)
