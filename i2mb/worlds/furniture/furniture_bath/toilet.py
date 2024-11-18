import numpy as np
from i2mb.worlds.furniture.base_furniture import BaseFurniture

from matplotlib.patches import Rectangle, Arc, PathPatch
from matplotlib.path import Path


class Toilet(BaseFurniture):
    width = 0.6
    height = 0.75

    def __init__(self, rotation=0, origin=None, scale=1):
        super().__init__(Toilet.height, Toilet.width, origin, rotation, scale)
        self.sitting_pos = self.dims / 2
        self.points.append(self.sitting_pos)

        # Tank
        self.tank_points = [np.array([0.05, 0.]), self.dims * [0.9, 0.2]]
        self.points.extend(self.tank_points)

        # Lid
        path_origin = np.array([0.0, self.height - self.width * 0.5])
        second_point = np.array([0.0, self.height * 0.25])
        third_point = np.array([self.width, self.height * 0.25])
        fourth_point = np.array([self.width, self.height - self.width * 0.5])
        self.lid_points = [path_origin, second_point, third_point, fourth_point]
        self.points.extend(self.lid_points)

        self.lead_center = np.array([self.width * 0.5, self.height - self.width * 0.5])
        self.points.append(self.lead_center)

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        if bbox:
            ax.add_patch(Rectangle(abs_origin, self.width, self.height, fill=True, facecolor="blue",
                                   linewidth=1.2, alpha=0.4, edgecolor="gray"))

        # Tank
        origin = self.tank_points[0]
        width, height = self.tank_points[1] - origin
        ax.add_patch(
            Rectangle(abs_origin + origin, width, height, fill=False, facecolor="gray",
                      linewidth=1.2, alpha=0.4, edgecolor="gray"))

        # Lid
        width = min(self.dims)
        ax.add_patch(Arc(abs_origin + self.lead_center, width, width,
                         angle=self.rotation, theta1=0, theta2=180, fill=False, linewidth=1.2,
                         edgecolor='gray', alpha=0.4))

        ax.add_patch(PathPatch(Path(abs_origin + self.lid_points),
                               fill=False, linewidth=1.2, edgecolor='grey', alpha=0.4))
