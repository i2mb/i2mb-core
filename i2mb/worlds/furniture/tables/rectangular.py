from itertools import product

import numpy as np
from matplotlib.patches import Rectangle

from ._table import MINIMUM_CHAIR_SPACE, BaseTable


class RectangularTable(BaseTable):
    def __init__(self, sits=6, horizontal=False, origin=None, **kwargs):
        super().__init__(1, 1, origin=origin, sits=sits, **kwargs)

        self.horizontal = horizontal

        reg_length1 = self.reg_length
        reg_width = self.reg_width
        self.width = (reg_length1 - MINIMUM_CHAIR_SPACE) * 2
        self.length = self.width

        # Determine table size:
        side_seats = 1
        if sits <= 4:
            sits = 4

        # Reduce head depth while maintaining effective eating area.
        eating_depth = (reg_length1 - MINIMUM_CHAIR_SPACE)
        eating_area = eating_depth * reg_width
        head_depth = eating_area / (eating_depth * 2)

        if sits > 4:
            side_seats = (sits - 2 + sits % 2) // 2
            self.length = side_seats * reg_width + head_depth * 2

        if horizontal:
            # Head of the table
            self._sitting_positions[-2, :] = [MINIMUM_CHAIR_SPACE * 0.7,
                                              MINIMUM_CHAIR_SPACE + self.width / 2]
            # Foot of the table
            self._sitting_positions[-1, :] = [MINIMUM_CHAIR_SPACE * 1.3 + self.length,
                                              MINIMUM_CHAIR_SPACE + self.width / 2]
            # Sides of the table
            side_w_pos = [MINIMUM_CHAIR_SPACE + head_depth + (reg_width / 2) + (reg_width * s) for s in
                          range(side_seats)]
            side_l_pos = [MINIMUM_CHAIR_SPACE * 0.7, MINIMUM_CHAIR_SPACE * 1.3 + self.width]
            self._sitting_positions[:-2, :] = np.array(list(product(side_l_pos, side_w_pos)))[:, ::-1]

        else:
            # Head of the table
            self._sitting_positions[-2, :] = [MINIMUM_CHAIR_SPACE + self.width / 2,
                                              MINIMUM_CHAIR_SPACE * 1.3 + self.length]
            # Foot of the table
            self._sitting_positions[-1, :] = [MINIMUM_CHAIR_SPACE + self.width / 2,
                                              MINIMUM_CHAIR_SPACE * 0.7]
            # Sides of the table
            side_l_pos = [MINIMUM_CHAIR_SPACE + head_depth + reg_width / 2 + reg_width * s for s in range(side_seats)]
            side_w_pos = [MINIMUM_CHAIR_SPACE * 0.7, MINIMUM_CHAIR_SPACE * 1.3 + self.width]
            self._sitting_positions[:-2, :] = np.array(list(product(side_l_pos, side_w_pos)))[:, ::-1]

    def get_bbox(self):
        bbox = [0, 0, 0, 0]
        bbox[0], bbox[1] = self.origin
        bbox[2] = self.width + 2 * MINIMUM_CHAIR_SPACE
        bbox[3] = self.length + 2 * MINIMUM_CHAIR_SPACE
        if self.horizontal:
            bbox[3] = self.width + 2 * MINIMUM_CHAIR_SPACE
            bbox[2] = self.length + 2 * MINIMUM_CHAIR_SPACE

        return bbox

    def draw(self, ax, bbox=True):
        width, length = self.width, self.length
        if self.horizontal:
            width, length = self.length, self.width

        ax.add_patch(Rectangle(self.origin + [MINIMUM_CHAIR_SPACE, MINIMUM_CHAIR_SPACE], width, length,
                               fill=True, facecolor="gray", linewidth=1.2,
                               alpha=0.4,
                               edgecolor='black'))

        if bbox:
            w, l = self.get_bbox()[-2:]
            ax.add_patch(Rectangle(self.origin, w, l,
                                   fill=False, facecolor="gray", linewidth=1.2,
                                   alpha=0.4,
                                   edgecolor='blue'))
