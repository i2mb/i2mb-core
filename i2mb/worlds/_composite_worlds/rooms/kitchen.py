from functools import partial
from random import uniform, randint

import numpy as np

import i2mb.activities.activity_descriptors
from i2mb.utils import global_time
from i2mb.utils.distributions import TemporalLinkedDistribution
from i2mb.worlds.furniture.kitchenUnit import KitchenUnit
from ._room import BaseRoom

"""
    :param outline: Shape of the kitchen unit. Possible choices are U, L and I.
    :type outline: char, optional
"""


class Kitchen(BaseRoom):
    def __init__(self, outline='U', dims=(3, 3.5), scale=1, **kwargs):
        super().__init__(dims=dims, scale=scale, **kwargs)

        # create kitchen unit
        if outline != 'U' and outline != 'L' and outline != 'I':
            print(str(outline) + " is not a valid shape, default is used.")
            outline = 'U'

        self.kitchen_unit = KitchenUnit(shape=outline, width=self.width,
                                        height=self.height, depth=0.75, scale=scale)

        self.add_furniture([self.kitchen_unit])

        n = 1
        if outline == "L":
            n = 2

        if outline == "U":
            n = 3

        self.outline = outline
        self.furniture_upper = np.empty((n, 2))
        self.furniture_origins = np.empty((n, 2))
        self.get_furniture_grid(outline)

        on_distribution = partial(np.random.choice, np.arange(1, 5), p=[0.6, 0.25, 0.10, 0.05])
        tld = TemporalLinkedDistribution(on_distribution, global_time.make_time(minutes=10))
        self.local_activities.extend([
            i2mb.activities.activity_descriptors.KitchenWork(
                location=self,
                duration=tld.sample_on,
                blocks_for=tld.sample_off,
                device=self.kitchen_unit)])

    def get_furniture_grid(self, outline):  # TODO comment not really origins
        k = self.kitchen_unit
        n = len(self.furniture_origins)
        offset = [0, 0, k.depth, k.width]
        self.furniture_origins[0] = [k.origin[0] - offset[(int(k.rotation / 90) + 1) % 4],
                                     k.origin[1] - offset[(int(k.rotation / 90)) % 4]]
        self.furniture_upper[0] = [k.origin[0] + offset[(int(k.rotation / 90) + 3) % 4],
                                   k.origin[1] + offset[(int(k.rotation / 90) + 2) % 4]]
        offset = [0, 0, k.width, k.depth]
        if n > 1:
            self.furniture_origins[1] = [k.origin[0] - offset[(int(k.rotation / 90) + 1) % 4],
                                         k.origin[1] - offset[(int(k.rotation / 90)) % 4]]
            self.furniture_upper[1] = [k.origin[0] + offset[(int(k.rotation / 90) + 3) % 4],
                                       k.origin[1] + offset[(int(k.rotation / 90) + 2) % 4]]
        if n == 3:
            rot = np.radians(k.rotation)
            origin = np.array([k.origin[0] + np.cos(rot) * (k.width - k.depth),
                               k.origin[1] + np.sin(rot) * (k.width - k.depth)])

            self.furniture_origins[2] += [origin[0] - offset[(int(k.rotation / 90) + 1) % 4],
                                          origin[1] - offset[(int(k.rotation / 90)) % 4]]
            self.furniture_upper[2] = [origin[0] + offset[(int(k.rotation / 90) + 3) % 4],
                                       origin[1] + offset[(int(k.rotation / 90) + 2) % 4]]

    def move(self, n):
        # move around along kitchen unit -> get borders (0.1 away for nicer visual)
        rot = np.radians(self.rotation)
        dim_x = abs(np.cos(rot) * self.dims[0] + np.sin(rot) * self.dims[1])
        dim_y = abs(np.sin(rot) * self.dims[0] + np.cos(rot) * self.dims[1])
        x0, x1 = 0.1, dim_x - 0.1
        y0, y1 = dim_y - self.kitchen_unit.depth - 0.1, dim_y - self.kitchen_unit.depth - 0.1

        if self.outline != "I":
            if self.rotation == 180 or self.rotation == 270:
                x1 = dim_x - self.kitchen_unit.depth - 0.1
            else:
                x0 = dim_x - self.kitchen_unit.depth - 0.1
            y0 = dim_y - self.kitchen_unit.length + 0.1
        if self.outline == "U":
            x0 = self.kitchen_unit.depth + 0.1
            x1 = dim_x - self.kitchen_unit.depth - 0.1
        # set x and y to somewhere along the working edge
        x = np.array([[uniform(x0, x1)] for i in range(n)])
        y = np.array([[uniform(y0, y1)] for i in range(n)])
        # choose if either on the vertical (~choice_x) or horizontal (choice_x) edge
        if self.outline != "I":
            choice_x = np.array([randint(0, 1) for i in range(n)], dtype=bool)
            if choice_x.any():
                y[choice_x] = y1
            if ~choice_x.any():
                x[~choice_x] = x1 if self.rotation == 180 or self.rotation == 270 else x0
                if self.outline == "U":
                    choice_left = np.array([randint(0, 1) for i in range((~choice_x).sum())], dtype=bool)
                    x[~choice_x & choice_left] = x0

        if self.rotation == 90 or self.rotation == 270:
            x, y = y, x
        return np.concatenate((x, y), axis=1)

    def start_activity(self, idx, descriptor_ids):
        bool_ix = self.population.find_indexes(idx)
        self.population.motion_mask[bool_ix] = True
        self.population.position[bool_ix] = self.dims/2
