from collections import deque

import numpy as np
from matplotlib.patches import Rectangle

from i2mb.activities.activity_descriptors import EatAtRestaurant
from i2mb.worlds import CompositeWorld
from i2mb.worlds.furniture.tables import RectangularTable
from i2mb.worlds.world_base import PublicSpace


class Restaurant(CompositeWorld, PublicSpace):
    def __init__(self, num_tables=10, reject_party=True, h_tables=False, seats_table=6, tables_per_row=4,
                 corridor_width=0.3, **kwargs):
        self.table_assignment = {}
        self.corridor_width = corridor_width
        self.tables_per_row = tables_per_row
        self.num_tables = num_tables
        self.reject_party = reject_party
        self.seats_table = seats_table
        self.sits = seats_table * num_tables
        # self.seats = np.zeros((num_tables * seats_table, 2))
        self.tables = [RectangularTable(sits=seats_table, horizontal=h_tables) for _ in range(num_tables)]
        self.available_tables = deque(self.tables, )

        # Determine restaurant dimensions
        tw, tl = self.tables[0].get_bbox()[2:]
        num_rows = int(np.ceil(num_tables / tables_per_row))
        w = (tables_per_row + 1) * corridor_width + tables_per_row * tw
        l = (tables_per_row + 1) * corridor_width + num_rows * tl
        kwargs["dims"] = (w, l)
        super().__init__(**kwargs)

        activities = [EatAtRestaurant(location=self)]
        self.local_activities.extend(activities)
        self.default_activity = activities[0]

        # Standalone building
        self.available_activities.extend(activities)

    def arrange_tables(self):
        row = col = 0
        t_width = t_length = 0
        for table in self.tables:
            table.origin = self.origin + [self.corridor_width + col * (t_width + self.corridor_width),
                                          self.corridor_width + row * (t_length + self.corridor_width)]
            t_width, t_length = table.get_bbox()[-2:]
            col += 1
            if col == self.tables_per_row:
                col = 0
                row += 1

    @CompositeWorld.origin.setter
    def origin(self, value):
        CompositeWorld.origin.fset(self, value)
        self.arrange_tables()

    def _draw_world(self, ax, bbox=False, origin=(0., 0.), **kwargs):
        ax.add_patch(Rectangle(self.origin, *self.dims, fill=False, linewidth=1.2, edgecolor='gray'))

        for table in self.tables:
            table.draw(ax, bbox=False)

    def available_places(self):
        return len(self.available_tables) * self.seats_table

    def can_sit_party(self, idx):
        return self.available_places() > len(idx)

    def sit_agents(self, idx):
        bool_idx = (self.population.index.reshape(-1, 1) == idx).any(axis=1)
        new_idx = np.arange(len(self.population))[bool_idx]
        num_tables = int(np.ceil(len(idx) / self.seats_table))
        start = 0
        end = self.seats_table
        for t in range(num_tables):
            table = self.available_tables.popleft()
            seats = len(idx[start:start + end])
            table.occupants += seats
            self.population.position[new_idx[start:start + end]] = table.get_sitting_positions()[:seats] - self.origin
            self.table_assignment.update(dict.fromkeys(idx[start:start + end], table))
            start += end

    def enter_world(self, n, idx=None, arriving_from=None):
        if hasattr(self.population, "motion_mask"):
            self.population.motion_mask[:] = False

        return np.zeros((n, 2))

    def exit_world(self, idx, global_population):
        bool_idx = (self.population.index.reshape(-1, 1) == idx).any(axis=1)
        if hasattr(self.population, "motion_mask"):
            self.population.motion_mask[bool_idx] = True

        super().exit_world(idx, global_population)

        for ix in idx:
            table = self.table_assignment.get(ix)
            if table is None:
                continue

            table.occupants -= 1
            del self.table_assignment[ix]
            if table.occupants == 0:
                self.available_tables.append(table)
