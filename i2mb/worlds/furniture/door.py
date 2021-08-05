import numpy as np
from i2mb.worlds.furniture.base_furniture import BaseFurniture
from matplotlib.patches import PathPatch, Arc, Rectangle
from matplotlib.path import Path

DOOR_WIDTH = 0.8


class Door(BaseFurniture):
    def __init__(self, rotation, origin, scale=1):
        self.door_width = DOOR_WIDTH * scale
        height = width = self.door_width
        super().__init__(height, width, origin, rotation, scale)
        self.door_begins = [self.door_width, 0]
        self.end = [self.door_width, self.door_width]

        self.points.extend([self.door_begins, self.end])

    def draw(self, ax, bbox=True, origin=(0, 0)):
        # self.door_begins, self.end = self.points
        abs_origin = self.door_begins + self.origin + origin
        abs_end = self.end + origin + self.origin

        rot = np.radians(self.rotation + 45)
        offset = np.array([self.door_width * -np.sin(rot),
                           self.door_width * np.cos(rot)])

        if bbox:
            ax.add_patch(Rectangle(self.origin + origin, self.width, self.height,
                                   fill=True, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="blue"))

        ax.add_patch(PathPatch(Path([abs_origin, offset + abs_origin]), fill=False, linewidth=1.2,
                               edgecolor='gray'))
        ax.add_patch(Arc(abs_origin, 2 * self.door_width, 2 * self.door_width, 0, (self.rotation + 90),
                         (self.rotation + 135), fill=False, linewidth=0.6, edgecolor='gray'))
