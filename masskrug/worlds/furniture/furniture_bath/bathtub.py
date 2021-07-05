import numpy as np

from matplotlib.patches import Rectangle, Arc, PathPatch
from matplotlib.path import Path


class Bathtub:
    def __init__(self, rotation=0, origin=None, scale=1, width=1, length=2):
        self.width = width * scale
        self.length = length * scale
        self.__origin = origin
        self.rotation = rotation
        if origin is None:
            self.__origin = np.array([0, 0])
        rot = np.radians(self.rotation)
        self.showering_pos = np.array([(np.cos(rot) * self.width - np.sin(rot) * self.length) / 2,
                                       (np.sin(rot) * self.width + np.cos(rot) * self.length) / 2])

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)
        self.showering_pos += self.__origin

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = origin + self.origin

        ax.add_patch(Rectangle(abs_origin, self.width, self.length, self.rotation,
                               fill=False, facecolor="gray", linewidth=1.2,
                               alpha=0.4, edgecolor="gray"))
        rot = np.radians(self.rotation)
        dist = self.width * 0.2
        dist_rot = [dist * (np.cos(rot) - np.sin(rot)) / 2, dist * (np.cos(rot) + np.sin(rot)) / 2]
        ax.add_patch(
            Arc((abs_origin[0] + (self.width / 2) * np.cos(rot) - np.sin(rot) * 3 * self.length / 4,
                 abs_origin[1] + (self.width / 2) * np.sin(rot) + np.cos(rot) * 3 * self.length / 4),
                self.width - dist,
                self.length / 2 - dist,
                180, 180 + self.rotation, 0 + self.rotation, fill=False, linewidth=1.2,
                edgecolor='gray', alpha=0.4))

        path_start = [-np.sin(rot) * (3 * self.length / 4 - dist / 2), np.cos(rot) * (3 * self.length / 4 - dist / 2)]
        path_stop = [np.cos(rot) * (self.width - dist), np.sin(rot) * (self.width - dist)]

        ax.add_patch(PathPatch(Path([(abs_origin + dist_rot), abs_origin + dist_rot + path_start]),
                               fill=False, linewidth=1.2, edgecolor='grey', alpha=0.4))
        ax.add_patch(PathPatch(Path([(abs_origin + dist_rot), abs_origin + dist_rot + path_stop]),
                               fill=False, linewidth=1.2, edgecolor='grey', alpha=0.4))
        ax.add_patch(PathPatch(Path([(abs_origin + dist_rot + path_stop),
                                     abs_origin + dist_rot + path_stop + path_start]),
                               fill=False, linewidth=1.2, edgecolor='grey', alpha=0.4))


class Shower:
    def __init__(self, rotation=0, origin=None, scale=1, width=1, length=1):
        self.width = width * scale
        self.length = length * scale
        self.__origin = origin
        self.rotation = rotation
        if origin is None:
            self.__origin = np.array([0, 0])
        rot = np.radians(self.rotation)
        self.showering_pos = np.array([(np.cos(rot) * self.width - np.sin(rot) * self.length) / 2,
                                       (np.sin(rot) * self.width + np.cos(rot) * self.length) / 2])

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)
        self.showering_pos += self.__origin

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        ax.add_patch(
            Rectangle(abs_origin, self.width, self.length, self.rotation, fill=False, facecolor="gray", linewidth=1.2,
                      alpha=0.4, edgecolor="gray"))
        rot = np.radians(self.rotation)
        ax.add_patch(PathPatch(
            Path([[abs_origin[0], abs_origin[1]],
                  [abs_origin[0] + np.cos(rot) * self.width - np.sin(rot) * self.length,
                   abs_origin[1] + np.sin(rot) * self.width + np.cos(rot) * self.length]]),
            fill=False, linewidth=1.2, edgecolor='gray', alpha=0.4))

        ax.add_patch(PathPatch(
            Path([[abs_origin[0] + np.cos(rot) * self.width - np.sin(rot) * self.length, abs_origin[1]],
                  [abs_origin[0], abs_origin[1] + np.sin(rot) * self.width + np.cos(rot) * self.length]]),
            fill=False, linewidth=1.2, edgecolor='gray', alpha=0.4))
