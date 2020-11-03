from itertools import product

from masskrug.worlds import CompositeWorld
import numpy as np


class Graveyard(CompositeWorld):
    def __init__(self, lots=30, lots_row=10, **kwargs):
        super().__init__(**kwargs)
        self.num_lots = lots
        self.current_lot = 0
        lot_width = self.dims[0] / lots_row
        x_pos = [p * lot_width + 0.5 * lot_width for p in range(lots_row)]

        num_rows = int(np.ceil(lots / lots_row))
        lot_height = self.dims[1] / num_rows
        y_pos = [p * lot_height + 0.5 * lot_height for p in range(num_rows)]

        self.lot_positions = np.array([c[::-1] for c in product(y_pos, x_pos)])
        self.lot_positions[:, 1] = self.dims[1] - self.lot_positions[:, 1]

    def step(self, t):
        self.population.motion_mask[:] = False

    def enter_world(self, n, idx=None):
        lots = self.lot_positions[self.current_lot: self.current_lot + n]
        self.current_lot += n
        return lots
