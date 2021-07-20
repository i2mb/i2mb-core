from masskrug.worlds import CompositeWorld
from matplotlib.patches import Rectangle, PathPatch, Arc
from matplotlib.path import Path
import numpy_indexed as npi
from random import uniform

import numpy as np

"""
    :param num_floors: Number of floors in the apartment building. Value above 1.
    :type num_floors: int, optional
    :param scale: percentage value for room scaling
    :type scale: float, optional
"""


class Stairs(CompositeWorld):
    def __init__(self, num_floors=4, scale=1, dims=(6, 28), **kwargs):
        super().__init__(dims=dims, **kwargs)
        self.dims = self.dims * scale
        self.num_floors = max(1, num_floors)
        self.scale = scale
        self.entry_point = [self.dims[0] / 4, 0.2]
        self.__room_entries = []
        self.__adjacent_rooms = []
        self.lift_points = [[0.85 * self.dims[0], i * self.dims[1] / num_floors + 0.2] for i in range(num_floors)]
        self.furniture_origins = None
        self.furniture_upper = None

    def get_room_entries(self):
        return self.__room_entries, self.__adjacent_rooms

    def set_room_entries(self, room_entry, id):
        self.__room_entries += room_entry
        self.__adjacent_rooms += id

    def enter_world(self, n, idx=None, arriving_from=None):
        if idx is None:
            return [self.entry_point] * n
        entries, rooms = np.array(self.__room_entries), np.array(self.__adjacent_rooms)
        indices = npi.indices(rooms, locations)
        return entries[indices]

    def exit_world(self, idx):
        bool_ix = self.population.find_indexes(idx)
        self.population.inter_target[bool_ix] = np.nan
        self.population.motion_mask[bool_ix] = True

    def draw_world(self, ax=None, origin=(0, 0), **kwargs):
        bbox = kwargs.get("bbox", False)
        self._draw_world(ax, origin=origin, **kwargs)
        for region in self.regions:
            region.draw_world(ax=ax, origin=origin + self.origin, **kwargs)

    def _draw_world(self, ax, bbox=False, origin=(0, 0), **kwargs):
        abs_origin = self.origin + origin
        ax.add_patch(
            Rectangle(self.origin + origin, *self.dims, fill=True, linewidth=1.2, edgecolor='blue', facecolor='blue',
                      alpha=0.2))
        width, height = self.dims[0], self.dims[1]
        ceil_height = height / self.num_floors

        # vertical separator in the middle
        path = [[width / 2, 0], [width / 2, height]]
        ax.add_patch(PathPatch(Path(abs_origin + path), fill=False, linewidth=1.2, edgecolor='gray'))
        # horizontal separators in between the floors
        for i in range(1, self.num_floors + 1):
            path = [[0, i * ceil_height], [width, i * ceil_height]]
            ax.add_patch(PathPatch(Path(abs_origin + path), fill=False, linewidth=1.2, edgecolor='gray'))

        # stairs pattern
        step = 0.5 * self.scale

        # other floors
        for i in range(self.num_floors - 1):
            for j in range(int(ceil_height / (2 * step)) + 1):
                d = step * j
                path1 = [[0, i * ceil_height + d], [width / 2, i * ceil_height + d]]
                path2 = [[width / 2, (i + 0.5) * ceil_height + d],
                         [width, (i + 0.5) * ceil_height + d]]
                ax.add_patch(
                    PathPatch(Path(abs_origin + path1), fill=False, linewidth=1.2, edgecolor='gray', alpha=0.4))
                ax.add_patch(
                    PathPatch(Path(abs_origin + path2), fill=False, linewidth=1.2, edgecolor='gray', alpha=0.4))

    def isInArray(self, array, value):
        for i in array:
            if i == value:
                return True
        return False

    def step(self, t):
        if not hasattr(self, "population"):
            return
        if not self.population:
            return

        # walk down
        width, height = self.dims[0], self.dims[1]
        x_pos_conc = np.array([[i[0]] for i in self.population.position])
        y_pos_conc = np.array([[i[1]] for i in self.population.position])
        x_pos = x_pos_conc.ravel()
        y_pos = y_pos_conc.ravel()
        floor_numbers = np.array([a.floor_number for a in self.population.home], dtype=int)

        floor_heights_y = np.array([i * height / self.num_floors + 0.75 for i in range(1, self.num_floors)])
        half_floor_heights_y = np.array([(i + 0.5) * height / self.num_floors for i in range(self.num_floors - 1)])
        floor_heights = [0.75]
        for i, j in zip(half_floor_heights_y, floor_heights_y):
            floor_heights += [i]
            floor_heights += [j]
        floor_heights = np.array(floor_heights)
        going_outside = self.population.target == self.entry_point
        at_entry_point = (x_pos == self.entry_point[0]) & (y_pos == self.entry_point[1])
        has_no_inter_target = np.isnan(self.population.inter_target)
        going_outside = np.array([all(i) for i in going_outside])
        has_no_inter_target = np.array([all(i) for i in has_no_inter_target])

        set_inter_target = going_outside & has_no_inter_target & ~at_entry_point & (y_pos != 0.75)
        if set_inter_target.any():
            # move horizontal
            move_right = np.array([self.isInArray(floor_heights_y, i) for i in y_pos]) & set_inter_target & ~(
                    (x_pos >= 4 / 6 * width) & (x_pos <= 5 / 6 * width))
            if move_right.any():
                x_positions = np.array([[uniform(4 / 6 * width, 5 / 6 * width)] for i in range(sum(move_right))])
                self.population.inter_target[move_right] = np.concatenate((x_positions, y_pos_conc[move_right]), axis=1)

            move_left = np.array([self.isInArray(half_floor_heights_y, i) for i in y_pos]) & set_inter_target & ~(
                    (x_pos >= 1 / 6 * width) & (x_pos <= 2 / 6 * width))
            if move_left.any():
                x_positions = np.array([[uniform(1 / 6 * width, 2 / 6 * width)] for i in range(sum(move_left))])
                self.population.inter_target[move_left] = np.concatenate((x_positions, y_pos_conc[move_left]), axis=1)
            # move vertical
            move_vertical = np.array([self.isInArray(floor_heights_y, i) for i in y_pos]) & set_inter_target & (
                    (x_pos >= 4 / 6 * width) & (x_pos <= 5 / 6 * width)) | \
                            np.array([self.isInArray(half_floor_heights_y, i) for i in y_pos]) & set_inter_target & (
                                    (x_pos >= 1 / 6 * width) & (x_pos <= 2 / 6 * width))
            if move_vertical.any():
                next_y = []
                for i in y_pos[move_vertical]:
                    next_y += [np.where(floor_heights == i)[0] - 1]
                next_y = np.array(next_y)
                self.population.inter_target[move_vertical] = np.concatenate(
                    (x_pos_conc[move_vertical], floor_heights[next_y]), axis=1)
        # walk up
        going_outside_lift = self.population.target == np.array(self.lift_points)[floor_numbers]
        going_outside_lift = np.array([all(i) for i in going_outside_lift])
        on_floor = y_pos == np.array([t[1] for t in self.population.target])
        using_lift = self.population.target == self.lift_points[0]
        using_lift = np.array([all(i) for i in using_lift])
        set_inter_target = ~on_floor & has_no_inter_target & ~using_lift & ~going_outside & ~going_outside_lift & (
                floor_numbers != 0)
        if set_inter_target.any():
            floor_heights[0] = self.entry_point[1]
            # move horizontal
            move_right = np.array([self.isInArray(half_floor_heights_y, i) for i in y_pos]) & set_inter_target & ~(
                    (x_pos >= 4 / 6 * width) & (x_pos <= 5 / 6 * width))
            if move_right.any():
                x_positions = np.array([[uniform(4 / 6 * width, 5 / 6 * width)] for i in range(sum(move_right))])
                self.population.inter_target[move_right] = np.concatenate((x_positions, y_pos_conc[move_right]), axis=1)

            move_left = np.array([self.isInArray(floor_heights_y, i) for i in y_pos]) & set_inter_target & ~(
                    (x_pos >= 1 / 6 * width) & (x_pos <= 2 / 6 * width))
            if move_left.any():
                x_positions = np.array([[uniform(1 / 6 * width, 2 / 6 * width)] for i in range(sum(move_left))])
                self.population.inter_target[move_left] = np.concatenate((x_positions, y_pos_conc[move_left]), axis=1)
            # move vertical
            move_vertical = (np.array([self.isInArray(half_floor_heights_y, i) for i in y_pos]) & set_inter_target & (
                    (x_pos >= 4 / 6 * width) & (x_pos <= 5 / 6 * width))) | \
                            (np.array([self.isInArray(floor_heights_y, i) for i in y_pos]) & set_inter_target & (
                                    (x_pos >= 1 / 6 * width) & (x_pos <= 2 / 6 * width)) | (
                                     y_pos == self.entry_point[1]))
            if move_vertical.any():
                next_y = []
                for i in y_pos[move_vertical]:
                    next_y += [np.where(floor_heights == i)[0] + 1]
                next_y = np.array(next_y)
                self.population.inter_target[move_vertical] = np.concatenate(
                    (x_pos_conc[move_vertical], floor_heights[next_y]), axis=1)
