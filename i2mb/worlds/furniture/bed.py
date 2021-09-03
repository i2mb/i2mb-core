import numpy as np
from matplotlib.patches import Rectangle

from i2mb.worlds.furniture.base_furniture import BaseFurniture


class Bed(BaseFurniture):
    def __init__(self, rotation=0, origin=None, scale=1, width=1, height=2):
        super().__init__(height=height, width=width, origin=origin, rotation=rotation, scale=scale)

        self.assignment = False
        self.sleeping_pos = self.dims/2
        self.points.extend([self.sleeping_pos])

        # Bed parts
        # Pillow
        self.pillow_origin = np.array([0.1, height - 0.45 - 0.1])
        self.pillow_opposite = np.array([0.9 * width, 2 - 0.1])
        self.points.extend([self.pillow_origin, self.pillow_opposite])

        # Cover
        self.cover_origin = np.array([0., 0.])
        self.cover_opposite = np.array([width, height - 0.45 - 0.2])
        self.points.extend([self.cover_origin, self.cover_opposite])

    def generate_bed_parts(self):
        self.points.extend([])

    def get_activity_position(self, origin=(0, 0), pos_id=None):
        return self.origin + self.sleeping_pos + origin

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        ax.add_patch(
            Rectangle(abs_origin, self.width, self.height, fill=True, facecolor="gray", linewidth=1.2,
                      alpha=0.2, edgecolor="gray"))

        # # pillow
        width, height = self.pillow_opposite - self.pillow_origin
        ax.add_patch(
            Rectangle(abs_origin + self.pillow_origin, width, height, fill=True,
                      facecolor="gray", linewidth=1.2, alpha=0.3, edgecolor="gray"))
        # # bed cover
        width, height = self.cover_opposite - self.cover_origin
        ax.add_patch(Rectangle(abs_origin + self.cover_origin, width, height,
                               fill=True, facecolor="gray",
                               linewidth=1.2, alpha=0.3, edgecolor="gray"))
