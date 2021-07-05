import numpy as np
from matplotlib.patches import Rectangle, Arc


class Bed:
    def __init__(self, rotation=0, origin=None, scale=1, width=1, length=2):
        self.width = width * scale
        self.length = length * scale
        self.__origin = origin
        self.rotation = rotation
        if origin is None:
            self.__origin = np.array([0, 0])
        self.assignment = False
        rot = np.radians(self.rotation)
        self.sleeping_pos = np.cos(rot) * self.width / 2 - np.sin(rot) * self.length / 2, \
                            np.sin(rot) * self.width / 2 + np.cos(rot) * self.length / 2

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)

    def get_sleeping_position(self, origin=(0, 0)):
        return self.origin + self.sleeping_pos + origin

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        ax.add_patch(
            Rectangle(abs_origin, self.width, self.length, self.rotation, fill=True, facecolor="gray", linewidth=1.2,
                      alpha=0.2, edgecolor="gray"))
        rot = np.radians(self.rotation)
        pad = 0.1 * self.width
        padding = [pad * (np.cos(rot) - np.sin(rot)), pad * (np.cos(rot) + np.sin(rot))]
        offset_cover = [(self.width / 2) * (-np.sin(rot)), (self.width / 2) * np.cos(rot)]

        # pillow
        ax.add_patch(
            Rectangle(abs_origin + padding, self.width - 2 * pad, (self.width - 2 * pad) / 2, self.rotation, fill=True,
                      facecolor="gray", linewidth=1.2, alpha=0.3, edgecolor="gray"))
        # bed cover
        ax.add_patch(Rectangle(abs_origin + padding + offset_cover, self.width - 2 * pad,
                               self.length - self.width / 2 - 2 * pad, self.rotation, fill=True, facecolor="gray",
                               linewidth=1.2, alpha=0.3, edgecolor="gray"))
