from itertools import product

import numpy as np
from matplotlib.patches import Rectangle, Polygon

from ._table import MINIMUM_CHAIR_SPACE, BaseTable


class Bar(BaseTable):
    def __init__(self, sits=8, origin=None, shape="L", bar_width=0.4, seats_main=4, **kwargs):
        super().__init__(bar_width, 2, origin=origin, sits=sits, **kwargs)
        self.seats_main = seats_main
        self.bar_width = bar_width
        self.shape = shape
        if shape not in ["L", "U", "|"]:
            raise RuntimeError(f"Wrong shape given for the bar ('{shape}'), only 'L', 'U', or '|' are allowed.")

        # Determine bar size
        sides = 1
        self.side_space = MINIMUM_CHAIR_SPACE
        if shape == "|":
            self.seats_main = seats_main = sits
            self.side_space = 0.1

        if shape == "U":
            sides = 2

        seats_side = sits - seats_main
        seats_per_side = int(np.ceil(seats_side / sides))

        self.width = self.reg_width * seats_main
        self.length = bar_width + seats_per_side * self.reg_width

        # Arrange main siting area

        main_pos_w = [self.side_space + self.reg_width / 2 + s * self.reg_width for s in range(seats_main)]
        main_pos_l = [MINIMUM_CHAIR_SPACE * 0.7]
        self._sitting_positions[:seats_main, :] = np.array(list(product(main_pos_l, main_pos_w)))[:, ::-1]

        # Arrange side sitting
        if seats_per_side > 0:
            offset = 0
            for side in range(sides):
                main_pos_w = [MINIMUM_CHAIR_SPACE * 0.7 + offset]
                main_pos_l = [MINIMUM_CHAIR_SPACE + self.reg_width / 2 + s * self.reg_width
                              for s in range(seats_per_side)]
                offset += MINIMUM_CHAIR_SPACE * 0.6 + self.width
                start = seats_main + side * seats_per_side
                end = seats_main + (side + 1) * seats_per_side
                self._sitting_positions[start:end, :] = np.array(list(product(main_pos_l, main_pos_w)))[:, ::-1]

    def get_bbox(self):
        bbox = [0, 0, 0, 0]
        bbox[0], bbox[1] = self.origin
        bbox[2] = self.width + self.side_space
        if self.shape == "L":
            bbox[2] += 0.1
        else:
            bbox[2] += self.side_space

        bbox[3] = self.length + MINIMUM_CHAIR_SPACE + 0.1
        return bbox

    def draw(self, ax, bbox=True):
        if self.shape == "|":
            path = [[0, 0],
                    [self.width, 0],
                    [self.width, self.bar_width],
                    [0, self.bar_width]]
        elif self.shape == "L":
            path = [[0, 0],
                    [self.width, 0],
                    [self.width, self.bar_width],
                    [self.bar_width, self.bar_width],
                    [self.bar_width, self.length],
                    [0, self.length]]
        else:
            path = [[0, 0],
                    [self.width, 0],
                    [self.width, self.length],
                    [self.width - self.bar_width, self.length],
                    [self.width - self.bar_width, self.bar_width],
                    [self.bar_width, self.bar_width],
                    [self.bar_width, self.length],
                    [0, self.length]]

        ax.add_patch(Polygon(self.origin + [self.side_space, MINIMUM_CHAIR_SPACE] + path,
                             fill=True, facecolor="gray", linewidth=1.2,
                             alpha=0.4,
                             edgecolor='black'))

        if bbox:
            w, l = self.get_bbox()[-2:]
            ax.add_patch(Rectangle(self.origin, w, l,
                                   fill=False, facecolor="gray", linewidth=1.2,
                                   alpha=0.4,
                                   edgecolor='blue'))
