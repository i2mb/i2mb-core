import numpy as np

# Reserved distance for a chair. The distance is measured outwards from the border of the table.
MINIMUM_CHAIR_SPACE = 0.7


class BaseTable:
    def __init__(self, origin=None, sits=6, reg_width=0.65, reg_length=1.10):
        self._sitting_positions = np.zeros((sits, 2))
        self.reg_length = reg_length
        self.reg_width = reg_width
        self.occupants = 0
        self.sits = sits
        self.__origin = origin
        if origin is None:
            self.__origin = np.array([0, 0])

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)

    def get_sitting_positions(self):
        return self._sitting_positions + self.__origin

    def draw(self, ax, bbox=True):
        pass
