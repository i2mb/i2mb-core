import numpy as np

# Reserved distance for a chair. The distance is measured outwards from the border of the table.
from masskrug.worlds.furniture.base_furniture import BaseFurniture

MINIMUM_CHAIR_SPACE = 0.7


class BaseTable(BaseFurniture):
    def __init__(self, height, width, origin=None, sits=6, reg_width=0.65, reg_length=1.10, rotation=0, scale=1):
        super().__init__(height, width, origin, rotation=rotation, scale=scale)
        self._sitting_positions = np.zeros((sits, 2))
        self.reg_length = reg_length
        self.reg_width = reg_width
        self.occupants = 0
        self.sits = sits

    def get_sitting_positions(self):
        return self._sitting_positions + self.origin

    def draw(self, ax, bbox=True):
        pass
