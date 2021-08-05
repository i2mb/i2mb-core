import numpy as np
from matplotlib.patches import Rectangle, Arc, PathPatch
from matplotlib.path import Path

class BusCabin:
    def __init__(self, origin=None, scale=1, width=1, length=2):
        self.width = width * scale
        self.length = length * scale
        self.__origin = origin


        if origin is None:
            self.__origin = np.array([0, 0])

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)

    def draw(self, ax, bbox=True):
        ax.add_patch(Rectangle(self.origin, self.width, self.length, fill=True, facecolor="gray", linewidth=1.2,
                      alpha=0.2, edgecolor="gray"))