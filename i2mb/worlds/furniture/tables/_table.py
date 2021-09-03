from itertools import product

import numpy as np

# Reserved distance for a chair. The distance is measured outwards from the border of the table.
from i2mb.worlds.furniture.base_furniture import BaseFurniture

MINIMUM_CHAIR_SPACE = 0.7


class BaseTable(BaseFurniture):
    def __init__(self, height, width, origin=None, sits=6, reg_width=0.65, reg_length=1.10, rotation=0, scale=1):
        super().__init__(height, width, origin, rotation=rotation, scale=scale)
        self._sitting_positions = np.zeros((sits, 2))
        self.reg_length = reg_length
        self.reg_width = reg_width
        self.occupants = 0
        self.sits = sits

        self.compute_sitting_positions()
        self.points.extend(self._sitting_positions)

    def compute_sitting_positions(self):
        # Determine table size:
        side_seats = 1
        if self.sits <= 4:
            self.sits = 4

        eating_depth = (self.reg_length - MINIMUM_CHAIR_SPACE)
        eating_area = eating_depth * self.reg_width
        head_depth = eating_area / (eating_depth * 2)

        if self.sits > 4:
            side_seats = (self.sits - 2 + self.sits % 2) // 2

        # Head of the table
        self._sitting_positions[0, :] = [-self.reg_width/2,
                                          self.height / 2]
        # Foot of the table
        self._sitting_positions[-1, :] = [self.reg_width/2 + self.width,
                                          self.height / 2]
        # Sides of the table
        side_w_pos = [head_depth + (self.width / side_seats * s) for s in range(side_seats)]
        side_l_pos = [-self.reg_length/4, self.reg_length/4 + self.height]
        positions = np.array(list(product(side_w_pos, side_l_pos)))
        self._sitting_positions[1:-1, :] = positions[positions[:, 0].argsort()]

    def get_sitting_positions(self):
        return self._sitting_positions + self.origin

    def get_activity_position(self, origin=(0, 0),  pos_id=None):
        seat = pos_id
        if pos_id is None:
            seat = np.random.choice(range(len(self._sitting_positions)), size=1)

        return self.get_sitting_positions()[seat]

    def draw(self, ax, bbox=True):
        pass
