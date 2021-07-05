import numpy as np

from matplotlib.patches import Rectangle, Arc


class CouchTable:
    def __init__(self, width=2, length=1, oval=False, rotation=0, origin=None):
        self.width = width
        self.length = length
        self.oval = oval
        self.rotation = rotation
        self.__origin = origin
        if origin is None:
            self.__origin = np.array([0, 0])

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        if not self.oval:
            ax.add_patch(Rectangle(abs_origin, self.width, self.length, self.rotation, fill=True, facecolor="gray",
                                   linewidth=1.2, alpha=0.4, edgecolor="gray"))
        else:
            ax.add_patch(Arc([abs_origin[0] + self.width / 2, abs_origin[1] + self.length / 2], self.width,
                             self.length, self.rotation, fill=True, facecolor="gray", linewidth=1.2, alpha=0.4,
                             edgecolor="gray"))
