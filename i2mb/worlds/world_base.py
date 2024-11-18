from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

import numpy as np

from i2mb.engine.model import Model
from i2mb.worlds._area import Area

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from i2mb.engine.agents import AgentList
    from i2mb.engine.relocator import Relocator


class World(Model, Area):
    def __init__(self, dims=None, height=None, width=None, origin=None, rotation=0, scale=1, subareas=None):
        Area.__init__(self, dims=dims, height=height, width=width, origin=origin, rotation=rotation, scale=scale,
                      subareas=subareas)
        self.landmarks = []

        # Add callable actions to perform when entering and existing a space
        self.entry_actions = []
        self.exit_actions = []

        # Define the logical levels an agent needs to traverse in order to exit/enter a building
        self.entry_route = np.array([self])

    def active_enter_world(self, n, idx=None, arriving_from=None):
        positions = self.enter_world(n, idx, arriving_from)
        for action in self.entry_actions:
            action(n, idx, self, arriving_from)

        return positions

    def active_exit_world(self, idx, global_population):
        self.exit_world(idx, global_population)
        for action in self.exit_actions:
            action(idx, self, global_population)

        return

    def is_empty(self):
        if hasattr(self, "population") and len(self.population) > 0:  # noqa
            return False

        return True

    def enter_world(self, n, idx=None, arriving_from=None):
        """The default agent place of entry is a randomly distributed event_location within the space.
        :param n:
        :param idx:
        :param arriving_from: Tells information where the agent is coming form.
        """
        return np.random.random((n, 2)) * self.dims

    def prepare_entrance(self, idx, global_population):
        """Adjust location state based on entering agents"""
        return

    def exit_world(self, idx, global_population):
        """Given a list of particle ids that left the world, adjust the world after their departure.
        :param global_population:
        """
        return

    def start_activity(self, idx, descriptor_ids):
        return

    def stop_activity(self, idx, descriptor_ids):
        return

    def available_places(self):
        """Return the number of available places, True if the region has no restriction on the number of
        occupants."""
        return True

    @abstractmethod
    def get_containment_regions(self, *args):
        pass

    @abstractmethod
    def get_home_regions(self, *args):
        pass

    @abstractmethod
    def draw_world(self, *args, **kwargs):
        pass

    @abstractmethod
    def check_positions(self, *args):
        pass

    def random_position(self, n):
        """Generates `n` random positions in the world."""
        return np.random.random((n, 2)) * self.dims

    def list_all_regions(self):
        return []


class Landmark(ABC):
    def __init__(self, world):
        self.world = world
        self.world.landmarks.append(self)

    @abstractmethod
    def remove_overlap(self):
        pass


class Scenario:
    """Scenario classes serve to consolidate world construction."""

    def __init__(self, population: 'AgentList', relocator: 'Relocator' = None, **kwargs):
        self.relocator = relocator
        self.v_space = 1
        self.h_space = 1
        self.left = 1
        self.right = 1
        self.top = 1
        self.bottom = 1

        self._world = None
        self.__num_particles = len(population)
        self.population = population

    @property
    def num_particles(self):
        return self.__num_particles

    @property
    def world(self):
        if self._world is None:
            self._build_world()

        return self._world

    @abstractmethod
    def _build_world(self):
        pass

    @abstractmethod
    def draw_world(self, ax=None):
        return ax

    @abstractmethod
    def dynamic_regions(self) -> List[World]:
        pass

    def compact_grid(self, grid):
        row_origin = np.array([self.left, self.bottom])
        for row in grid[::-1]:
            column_origin = row_origin.copy()
            row_top_right = np.array([0, 0])
            for cell in row:
                if not isinstance(cell, list):
                    cell = [cell]

                old_cell_origin = np.vstack([a.origin for a in cell]).min(axis=0).copy()
                for area in cell:
                    relative_origin = area.origin - old_cell_origin
                    area.origin = column_origin + relative_origin
                    row_top_right = np.vstack([row_top_right, area.origin + area.dims]).max(axis=0)

                column_origin[0] = row_top_right[0] + self.v_space

            row_origin[1] = row_top_right[1] + self.h_space

    def arrange_grid(self, grid, wrap_shape=None, offset=None):
        if offset is None:
            offset = [0, 0]

        offset = list(offset)

        used_height = self.bottom
        max_width = 0
        max_col_widths = defaultdict(lambda: 0)
        for row in grid[::-1]:
            w, h = self.arrange_column(row, wrap_shape, offset, max_col_widths)
            used_height += h + self.h_space
            offset[1] += h + self.h_space
            if w > max_width:
                max_width = w

        used_height += self.top - self.h_space

        return max_width, used_height

    def arrange_column(self, column, wrap_shape=None, offset=None, col_widths=None):
        if offset is None:
            offset = [0, 0]

        if col_widths is None:
            col_widths = defaultdict(lambda: 0)

        offset = list(offset)

        used_width = self.left
        offset[0] += used_width
        max_height = 0
        for col, cell in enumerate(column):
            if not cell:
                used_width += col_widths[col] + self.v_space
                offset[0] += col_widths[col] + self.v_space
                continue

            w, h = self.arrange_cell(cell, wrap_shape, offset)
            if h > max_height:
                max_height = h

            if w > col_widths[col]:
                col_widths[col] = w

            offset[0] += w + self.v_space
            used_width += w + self.v_space
        used_width += self.right - self.v_space
        return used_width, max_height

    def arrange_cell(self, cell, wrap_shape=None, offset=None):
        if offset is None:
            offset = [0, 0]

        if not isinstance(cell, list):
            cell = [cell]

        if wrap_shape is None:
            wrap_shape = len(cell)

        n_cols = wrap_shape
        n_rows = len(cell) // n_cols + (len(cell) % n_cols != 0 and 1 or 0)
        used_height = 0
        max_used_width = 0
        w_tier = iter(cell)
        for r in range(n_rows):
            max_height = 0
            used_width = 0
            for c in range(n_cols):
                try:
                    w = next(w_tier)
                except StopIteration:
                    break

                width, height = w.dims
                if height > max_height:
                    max_height = height

                w.origin = np.array([used_width + offset[0], used_height])
                used_width += width + self.v_space

            used_width -= self.v_space
            if max_used_width < used_width:
                max_used_width = used_width

            used_height += max_height + self.h_space

        used_height -= self.h_space
        # inverting the grid
        height_1st_row = cell[0].origin[1]
        for w in cell:
            if height_1st_row != w.origin[1]:
                height_1st_row = w.origin[1] - self.h_space
                break

        if height_1st_row == cell[0].origin[1]:
            height_1st_row = used_height

        for w in cell:
            origin = w.origin
            origin[1] = -1 * (w.origin[1] - used_height) - height_1st_row + offset[1]
            w.origin = origin

        return max_used_width, used_height


class BlankSpace(Area):
    def __init__(self, dims):
        super().__init__(dims)

    def is_empty(self):
        return True

    def draw_world(self, *args, **kwargs):
        return

    def post_init(self):
        return

    def list_all_regions(self):
        return [self]


class PublicSpace(World):
    pass


def ravel_grids(grid):
    for r in grid:
        try:
            iter(r)
            yield from ravel_grids(r)
        except TypeError:
            yield r
