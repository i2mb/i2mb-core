import numpy as np

from matplotlib.patches import Rectangle, Arc, PathPatch
from matplotlib.path import Path


class Toilet:
    def __init__(self, rotation=0, origin=None, scale=1, width=0.6, length=0.75):
        self.width = width * scale
        self.length = length * scale
        self.__origin = origin
        self.rotation = rotation
        rot = np.radians(rotation)
        self.sitting_pos = np.array([(np.cos(rot) * self.width - np.sin(rot) * self.length) / 2,
                                     (np.sin(rot) * self.width + np.cos(rot) * self.length) / 2])

        if origin is None:
            self.__origin = np.array([0, 0])

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)
        self.sitting_pos += self.__origin

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        rot = np.radians(self.rotation)
        ax.add_patch(
            Rectangle(abs_origin, self.width * 0.8, self.length * 0.2, self.rotation, fill=False, facecolor="gray",
                      linewidth=1.2, alpha=0.4, edgecolor="gray"))

        arc_offset = [np.cos(rot) * self.width * 0.4 - np.sin(rot) * self.length * 0.7,
                      np.sin(rot) * self.width * 0.4 + np.cos(rot) * self.length * 0.7]
        ax.add_patch(Arc((abs_origin[0] + arc_offset[0], abs_origin[1] + arc_offset[1]), self.width, self.width,
                         180, 180 + self.rotation, 0 + self.rotation, fill=False, linewidth=1.2,
                         edgecolor='gray', alpha=0.4))

        offset = [-np.sin(rot) * self.length * 0.2 - np.cos(rot) * self.width * 0.1,
                  np.cos(rot) * self.length * 0.2 - np.sin(rot) * self.width * 0.1]

        path_start = [-np.sin(rot) * self.length * 0.5, np.cos(rot) * self.length * 0.5]
        path_stop = [np.cos(rot) * self.width, np.sin(rot) * self.width]

        ax.add_patch(PathPatch(Path([abs_origin + offset, abs_origin + offset + path_start]),
                               fill=False, linewidth=1.2, edgecolor='grey', alpha=0.4))
        ax.add_patch(PathPatch(Path([abs_origin + offset, abs_origin + offset + path_stop]),
                               fill=False, linewidth=1.2, edgecolor='grey', alpha=0.4))
        ax.add_patch(PathPatch(Path([abs_origin + offset + path_stop, abs_origin + offset + path_stop + path_start]),
                               fill=False, linewidth=1.2, edgecolor='grey', alpha=0.4))
