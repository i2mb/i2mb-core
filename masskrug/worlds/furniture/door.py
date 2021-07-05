import numpy as np
from matplotlib.patches import PathPatch, Arc
from matplotlib.path import Path

DOOR_WIDTH = 0.8


class Door:
    def __init__(self, rotation, origin, scale=1):
        self.__origin = origin
        self.rotation = rotation
        self.door_width = DOOR_WIDTH * scale
        if origin is None:
            self.__origin = np.array([0, 0])
        self.end = [self.origin[0] + np.cos(np.radians(rotation)) * self.door_width,
                    self.origin[1] + np.sin(np.radians(rotation)) * self.door_width]

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        abs_end = [self.end[0] + origin[0], self.end[1] + origin[1]]
        ax.add_patch(PathPatch(Path([abs_origin, abs_end]), fill=False, linewidth=1.2, edgecolor='white'))
        rot = np.radians(self.rotation)
        offset = np.array([self.door_width * np.sqrt(0.5) * (np.cos(rot) + np.sin(rot)),
                           self.door_width * np.sqrt(0.5) * (-np.cos(rot) + np.sin(rot))])
        ax.add_patch(PathPatch(Path([abs_origin, abs_origin + offset]), fill=False, linewidth=1.2, edgecolor='gray'))
        ax.add_patch(Arc(abs_origin, 2 * self.door_width, 2 * self.door_width, 45, self.rotation + 270,
                         self.rotation + 315, fill=False, linewidth=0.6, edgecolor='gray'))
