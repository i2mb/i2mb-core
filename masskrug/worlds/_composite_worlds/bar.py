from collections import deque

import numpy as np

from matplotlib.patches import Rectangle

from masskrug.worlds import CompositeWorld
from masskrug.worlds.furniture.tables import RectangularTable
from masskrug.worlds.furniture.tables.bar import Bar as BarTable
from masskrug.worlds.world_base import PublicSpace


class Bar(CompositeWorld, PublicSpace):
    def __init__(self, bar_shape="L", corridor_width=0.3, **kwargs):
        self.table_assignment = {}
        self.corridor_width = corridor_width
        self.bar_shape = bar_shape
        self.bar = BarTable(sits=12, shape=bar_shape, seats_main=6)
        self.occupied_stools = np.zeros(12, dtype=bool)
        num_tables = 4
        num_r_tables = 0
        if bar_shape == "U":
            num_r_tables = 1

        if bar_shape == "L":
            num_r_tables = 2

        self.seats_table = 6
        self.lower_tables = [RectangularTable(sits=self.seats_table, ) for _ in range(num_tables)]
        self.right_tables = []
        if num_r_tables > 0:
            self.right_tables.extend(
                [RectangularTable(sits=self.seats_table, horizontal=True) for _ in range(num_r_tables)])

        self.available_tables = deque([], )
        self.available_tables.extend(self.lower_tables)
        self.available_tables.extend(self.right_tables)
        self.sits = (num_tables + num_r_tables) * self.seats_table + self.bar.sits

        # Compute Dimensions
        tw, th = self.lower_tables[0].get_bbox()[2:]
        bw, bh = self.bar.get_bbox()[2:]
        width = (num_tables + 1) * self.corridor_width + num_tables * tw
        length = 3 * self.corridor_width + bh + th
        print(f"Bar dimensions: {width} {length}")
        kwargs["dims"] = (width, length)
        super().__init__(**kwargs)

    def arrange_tables(self):
        row = col = 0
        t_width = t_length = 0
        for table in self.lower_tables:
            table.origin = self.origin + [self.corridor_width + col * (t_width + self.corridor_width),
                                          self.corridor_width]
            t_width, t_length = table.get_bbox()[-2:]
            col += 1

        if self.bar_shape == "L":
            b_width = self.bar.get_bbox()[2]
            col = col * (t_width + self.corridor_width) - b_width
            base_line = self.corridor_width + t_length
            self.bar.origin = self.origin + [col, base_line + self.corridor_width]
            col = 1.10
            for table in self.right_tables:
                table.origin = self.origin + [col,
                                              base_line + self.corridor_width + row * (t_length + self.corridor_width)]
                t_width, t_length = table.get_bbox()[-2:]
                row += 1

        else:
            col = col * (t_width + self.corridor_width) - t_length
            base_line = self.corridor_width + t_length
            for table in self.right_tables:
                table.origin = self.origin + [col,
                                              base_line + self.corridor_width + row * (t_length + self.corridor_width)]
                t_width, t_length = table.get_bbox()[-2:]
                row += 1

            self.bar.origin = self.origin + [1.10, base_line + self.corridor_width]

    @CompositeWorld.origin.setter
    def origin(self, value):
        CompositeWorld.origin.fset(self, value)
        self.arrange_tables()

    def _draw_world(self, ax, bbox=False):
        ax.add_patch(Rectangle(self.origin, *self.dims, fill=False, linewidth=1.2, edgecolor='gray'))
        self.bar.draw(ax, bbox)
        for table in self.lower_tables:
            table.draw(ax, bbox)

        for table in self.right_tables:
            table.draw(ax, bbox)

    def available_places(self):
        return len(self.available_tables) * self.seats_table + self.bar.occupants

    def sit_particles(self, idx):
        bool_idx = (self.population.index.reshape(-1, 1) == idx).any(axis=1)
        new_idx = np.arange(len(self.population))[bool_idx]
        if len(idx) <= 2 and self.bar.occupants + len(idx) <= self.bar.sits:
            stool_positions = self.bar.get_sitting_positions()
            stool_idx = np.array(range(len(stool_positions)))
            stools = np.random.choice(stool_idx[~self.occupied_stools],
                                      len(idx), replace=False)

            self.bar.occupants += len(idx)
            self.population.position[bool_idx] = stool_positions[stools, :] - self.origin
            self.table_assignment.update(dict.fromkeys(idx, self.bar))
            return

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
            if len(self.available_tables) == 0:
                break

    def can_sit_party(self, idx):
        if len(idx) <= 2 and self.bar.occupants + len(idx) <= self.bar.sits:
            return True

        return len(self.available_tables) * self.seats_table > len(idx)

    def enter_world(self, n, idx=None):
        if hasattr(self.population, "motion_mask"):
            self.population.motion_mask[:] = False

        return np.zeros((n, 2))

    def exit_world(self, idx):
        bool_idx = (self.population.index.reshape(-1, 1) == idx).any(axis=1)
        if hasattr(self.population, "motion_mask"):
            self.population.motion_mask[bool_idx] = True

        for ix in idx:
            table = self.table_assignment[ix]
            table.occupants -= 1
            del self.table_assignment[ix]
            if table.occupants == 0 and table != self.bar:
                self.available_tables.append(table)
