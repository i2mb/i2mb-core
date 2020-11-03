from itertools import product

import numpy as np

from masskrug.utils import global_time
from masskrug.worlds import CompositeWorld


class PartyRoom(CompositeWorld):
    def __init__(self, num_tables=5, duration=15, table_radius=0.7, always_on=False, **kwargs):
        CompositeWorld.__init__(self, **kwargs)
        self.table_radius = table_radius
        self.num_tables = num_tables
        self.tables = np.zeros((num_tables, 2))
        self.arrange_tables()

        if hasattr(self.population, "target"):
            self.target = self.population.target
        else:
            self.target = np.zeros((len(self.population), 2))
            self.population.add_property("target", self.target)

        self._duration = duration
        if isinstance(duration, int):
            self._duration = lambda x: np.ones(x) * duration

        self.duration = self._duration(len(self.population))
        self.arrival_time = np.zeros((len(self.population)))

        self.assign_tables(self.population.index)

    def arrange_tables(self):
        num_cols = int(np.ceil(np.sqrt(self.num_tables)))
        num_rows = int(np.ceil(self.num_tables / num_cols))
        cell_dims = np.array([(self.dims[0] / num_cols), (self.dims[1] / num_rows)])
        cell_center = cell_dims / 2
        for table, idxs in enumerate(product(range(num_cols), range(num_rows))):
            if table > self.num_tables:
                break

            col_idx, row_idx = idxs
            self.tables[table, :] = [col_idx * cell_dims[0] + cell_center[0],
                                     row_idx * cell_dims[1] + cell_center[1]]

    def assign_tables(self, idx):
        num_targets = len(idx)
        if idx.dtype == bool:
            num_targets = sum(idx)

        tables = np.random.choice(range(len(self.tables)), num_targets)
        chairs_angle = (np.random.random(len(tables)) * 2 * np.pi)
        chairs = np.array(list(zip(np.cos(chairs_angle) * self.table_radius, np.sin(chairs_angle) * self.table_radius)))
        self.target[idx, :] = self.tables[tables, :] - chairs

    def step(self, t):
        if not self.population:
            return

        n = len(self.population)

        need_change = (t - self.arrival_time) >= self.duration
        if not need_change.any():
            return

        self.assign_tables(need_change)
        self.arrival_time[need_change] = t
        self.duration[need_change] = self._duration(sum(need_change))
        return
