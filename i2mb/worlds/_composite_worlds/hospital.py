from collections import deque
from itertools import product

import numpy as np

from i2mb.pathogen import UserStates
from i2mb.worlds import CompositeWorld


class Hospital(CompositeWorld):
    def __init__(self, beds=30, beds_row=10, **kwargs):
        super().__init__(**kwargs)
        self.num_beds = beds
        self.available_beds = deque(list(range(beds)))
        self.bed_assignment = {}
        room_width = self.dims[0] / beds_row
        x_pos = [p * room_width + 0.5 * room_width for p in range(beds_row)]

        num_rows = int(np.ceil(beds / beds_row))
        room_height = self.dims[1] / num_rows
        y_pos = [p * room_height + 0.5 * room_height for p in range(num_rows)]

        self.bed_positions = np.array([c[::-1] for c in product(y_pos, x_pos)])
        self.bed_positions[:, 1] = self.dims[1] - self.bed_positions[:, 1]

    def enter_world(self, n, idx=None, arriving_from=None):
        self.population.remain[:] = True
        self.population.motion_mask[:] = False

        beds = []
        for p_ix in self.population.index:
            if p_ix in self.bed_assignment:
                continue

            bed = self.available_beds.pop()
            self.bed_assignment[p_ix] = bed
            beds.append(bed)

        bed_pos = self.bed_positions[beds]
        return bed_pos

    def exit_world(self, idx, global_population):
        for ix in idx:
            bed = self.bed_assignment[ix]
            self.available_beds.append(bed)
            del self.bed_assignment[ix]

        return

    def step(self, t):
        if not hasattr(self, "population"):
            return

        if hasattr(self.population, "state"):
            recovered = ((self.population.state == UserStates.immune) |
                         (self.population.state == UserStates.deceased)).ravel()

            if recovered.any():
                self.population.remain[recovered] = False
                self.population.motion_mask[recovered] = True
