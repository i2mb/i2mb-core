import numpy as np
from masskrug.worlds.furniture.base_furniture import BaseFurniture

from matplotlib.patches import Rectangle, Arc, PathPatch
from matplotlib.path import Path


class Bathtub(BaseFurniture):
    def __init__(self, height=2, rotation=0, origin=None, scale=1, width=1):
        super().__init__(height, width, origin, rotation, scale)

        self.showering_pos = self.dims / 2
        self.points.extend([self.showering_pos])

        # Inner wall
        self.recess = 0.15
        path_origin = np.array([self.recess, self.height - (self.width - 0.2) * 0.5])
        second_point = np.array([self.recess, self.recess])
        third_point = np.array([self.width - self.recess, self.recess])
        fourth_point = np.array([self.width - self.recess, self.height - (self.width - self.recess) * 0.5])
        self.inner_wall_points = [path_origin, second_point, third_point, fourth_point]
        self.points.extend(self.inner_wall_points)

        self.lead_center = np.array([self.width * 0.5, self.height - (self.width - self.recess) * 0.5])
        self.points.append(self.lead_center)

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = origin + self.origin

        ax.add_patch(Rectangle(abs_origin, self.width, self.height,
                               fill=False, facecolor="gray", linewidth=1.2,
                               alpha=0.4, edgecolor="gray"))
        rot = np.radians(self.rotation)
        dist = self.width * 0.2
        dist_rot = [dist * (np.cos(rot) - np.sin(rot)) / 2, dist * (np.cos(rot) + np.sin(rot)) / 2]
        width = min(self.dims)
        ax.add_patch(
            Arc(abs_origin + self.lead_center,
                width - (2 * self.recess),
                width - (2 * self.recess),
                self.rotation, 0, 180, fill=False, linewidth=1.2,
                edgecolor='gray', alpha=0.4))

        ax.add_patch(PathPatch(Path(abs_origin + self.inner_wall_points),
                               fill=False, linewidth=1.2, edgecolor='grey', alpha=0.4))


class Shower(BaseFurniture):
    def __init__(self, rotation=0, origin=None, scale=1, width=1, height=1):
        super().__init__(height, width, origin, rotation, scale)

        self.showering_pos = self.dims / 2
        self.points.extend([self.showering_pos])

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        ax.add_patch(
            Rectangle(abs_origin, self.width, self.height, fill=False, facecolor="gray", linewidth=1.2,
                      alpha=0.4, edgecolor="gray"))

        ax.add_patch(PathPatch(
            Path([abs_origin, abs_origin + self.dims]),
            fill=False, linewidth=1.2, edgecolor='gray', alpha=0.4))

        ax.add_patch(PathPatch(
            Path([[abs_origin[0], (abs_origin + self.dims)[1]],
                  [(abs_origin + self.dims)[0], abs_origin[1]]]),
            fill=False, linewidth=1.2, edgecolor='gray', alpha=0.4))
