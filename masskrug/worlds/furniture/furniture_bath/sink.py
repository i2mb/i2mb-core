import numpy as np

from matplotlib.patches import Rectangle


class Sink:
    def __init__(self, rotation=0, origin=None, width=0.5, length=0.7, scale=1):
        self.width = width * scale
        self.length = length * scale
        self.__origin = origin
        self.rotation = rotation
        if origin is None:
            self.__origin = np.array([0, 0])

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = origin + self.origin

        offset = 0.1 * self.width
        rot = np.radians(self.rotation)
        offset_rot = [offset * (np.cos(rot) - np.sin(rot)), offset * (np.cos(rot) + np.sin(rot))]

        ax.add_patch(Rectangle(abs_origin, self.width, self.length, self.rotation,
                               fill=False, facecolor="gray", linewidth=1.2,
                               alpha=0.4, edgecolor="gray"))
        ax.add_patch(Rectangle((abs_origin + offset_rot), self.width - offset * 2, self.length - offset * 2,
                               self.rotation, fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))
