import numpy as np
from itertools import product
from matplotlib.patches import Rectangle

from ._table import BaseTable


class DiningTable(BaseTable):
    def __init__(self, sits=8, rotation=0, height=1, width=2, scale=1, origin=None, **kwargs):
        super().__init__(height=height, width=width, origin=origin, sits=sits, rotation=rotation, scale=scale, **kwargs)

        # head of table
        # self._sitting_positions[-2, :] = [-np.sin(rot) * self.height / 2,
        #                                   np.cos(rot) * self.height / 2]
        #
        # # foot of table
        # self._sitting_positions[-1, :] = [- np.sin(rot) * self.height / 2 + np.cos(rot) * self.width,
        #                                   np.cos(rot) * self.height / 2 + np.sin(rot) * self.width]
        #
        # # sides of table
        # side_seats = (sits - 2 + sits % 2) // 2
        # chairspace = self.width / side_seats
        # side_w_pos = [(chairspace / 2 + chairspace * s) for s in range(side_seats)]
        #
        # if self.rotation == 180 or self.rotation == 270:
        #     side_w_pos = [i * -1 for i in side_w_pos]
        #
        # side_l_pos = [-np.sin(rot) * self.height,
        #               np.cos(rot) * self.height]
        #
        # self._sitting_positions[:-2, :] = np.array(list(product(side_l_pos, side_w_pos)))[:, ::-1]

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        ax.add_patch(Rectangle(abs_origin, self.width, self.height,
                               fill=True, facecolor="gray", linewidth=1.2,
                               alpha=0.4,
                               edgecolor='gray'))
