#  dct_mct_analysis
#  Copyright (C) 2021  FAU - RKI
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from matplotlib.patches import Rectangle

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..engine.agents import AgentList

from .world_base import World
from ..utils import cache_manager


class CompositeWorld(World):
    def __init__(self, dims=None, population: 'AgentList' = None, regions=None, origin=None, map_file=None,
                 containment=False, waiting_room=False, rotation=0, scale=1):

        self.regions = []
        super().__init__(dims, origin=origin, rotation=rotation, scale=scale, subareas=self.regions)

        self.waiting_room = waiting_room
        self.containment = containment

        # Region description
        self.region_origins = np.array([])
        self.region_dimensions = np.array([])

        # Handle Single index with views
        self.__region_index = np.array([[-1, 0, -1],
                                       [self.id, 0, self]])
        self.__region_index_slice = slice(None)
        self.__region_index_pos = 1
        self.__blocked_locations = np.array([False, False], dtype=bool)

        # Activity Information
        self.local_activities = []
        self.available_activities = []
        self.activity_types = set()

        # Define the logical levels an agent needs to traverse in order to exit a building
        self.entry_route = np.array([self])

        if regions:
            self.add_regions(regions)

        if map_file is not None:
            self.load_map(map_file)

        self.population = population
        if population is None:
            self.population = None
            self.gravity = np.zeros((0, 2))
            self.containment_region = np.array([])
            if waiting_room:
                self.in_waiting_room = np.array([])

            return

        n = len(population)
        self.__in_region = np.zeros(n, dtype=bool)
        self.containment_region = np.empty((n,), dtype=object)
        self.home = np.empty((n,), dtype=object)
        self.at_home = np.zeros((n,), dtype=bool)

        population.add_property("containment_region", self.containment_region)
        population.add_property("home", self.home)
        population.add_property("at_home", self.at_home)

        self.gravity = np.zeros((n, 2))
        population.add_property("gravity", self.gravity)

        if waiting_room:
            self.in_waiting_room = np.zeros((n, 1), dtype=bool)
            population.add_property("in_waiting_room", self.in_waiting_room)

    def __hash__(self):
        return hash(id(self))

    @property
    def region_index(self):
        return self.__region_index[self.__region_index_slice]

    @property
    def blocked(self):
        return self.__blocked_locations[self.__region_index_pos]

    @blocked.setter
    def blocked(self, v):
        self.__blocked_locations[self.__region_index_pos] = v

    @property
    def index(self):
        return self.__region_index_pos

    @property
    def blocked_locations(self):
        return self.__blocked_locations[self.__region_index_slice]

    def block_locations(self, selector, v):
        idx = np.arange(len(self.__blocked_locations), dtype=int)
        idx = idx[self.__region_index_slice][selector]
        self.__blocked_locations[idx] = v

    def check_positions(self, idx):
        if self.is_empty():
            return

        mask = self.population.find_indexes(idx).ravel()
        positions = self.population.position[mask]
        self.population.position[mask] = self.constrain_positions(positions)
        for lm in self.landmarks:
            lm.remove_overlap()

    def constrain_positions(self, positions):
        check_top = (positions[:, 0] > self.dims[0])
        check_right = (positions[:, 1] > self.dims[1])
        check_left = (positions[:, 0] < 0)
        check_bottom = (positions[:, 1] < 0)

        # enforce constraints
        positions[check_top, 0] = self.dims[0]
        positions[check_right, 1] = self.dims[1]
        positions[check_left, 0] = 0
        positions[check_bottom, 1] = 0
        return positions

    def get_containment_regions(self, idx):
        return self.containment_region[idx]

    def get_home_regions(self, idx):
        return self.home[idx]

    def load_map(self, map_file):
        pass

    def add_regions(self, regions):
        self.regions.extend(regions)
        for region in regions:
            region.parent = self

        self.update_region_index()

        # Adjust geometries
        self.region_origins = np.array([r.origin for r in self.regions])
        self.region_dimensions = np.array([r.dims for r in self.regions])

        region_origin = np.min(self.region_origins, axis=0)
        if (region_origin < [0., 0.]).any():
            mask = region_origin < [0., 0.]
            for r in self.regions:
                r.origin += self.origin - (region_origin * mask)

        self.points.extend([p for r in regions for p in [r.origin, r.opposite]])

        region_dimensions = np.max(self.region_dimensions + self.region_origins, axis=0)
        self.dims = region_dimensions

    def update_region_index(self):
        region_index = [[-1, 0, -1]]
        for region in self.list_all_regions():
            region_index.append([region.id, 0, region])

        self.__region_index = np.array(region_index)
        sort = np.argsort(self.__region_index[:, 0])
        self.__region_index = self.__region_index[sort]
        self.__region_index_pos = np.searchsorted(self.__region_index[:, 0], self.id)
        self.__blocked_locations = np.zeros(len(self.__region_index), dtype=bool)

        # Unify index and blocking across all regions
        self.unify_index()

        # replace parent.id for actual parent index
        self.__region_index[:, 1] = [r != -1 and
                                     (r.parent is not None and r.parent.index or 0)
                                     or 0 for r in self.__region_index[:, 2]]

    def unify_index(self):
        for region in self.list_all_regions():
            if region == self:
                continue

            region.__region_index = self.__region_index
            region.__blocked_locations = self.__blocked_locations
            region.__region_index_pos = np.searchsorted(self.__region_index[:, 0], region.id)
            region.__region_index_slice = np.sort(
                np.searchsorted(self.__region_index[:, 0], [-1] + [r.id for r in region.list_all_regions()]))

    def get_absolute_origin(self):
        if self.parent is None:
            return self.origin

        return self.origin + self.parent.get_absolute_origin()

    def draw_world(self, ax=None, origin=(0, 0), **kwargs):
        bbox = kwargs.get("bbox", False)
        self._draw_world(ax, origin=self.origin + origin, **kwargs)
        for region in self.regions:
            region.draw_world(ax=ax,  origin=self.origin + origin, **kwargs)

    def _draw_world(self, ax, bbox=False, origin=(0, 0), **kwargs):
        ax.add_patch(Rectangle(origin, *self.dims, fill=False, linewidth=1.2, edgecolor='gray'))

    def get_entrance_sub_region(self):
        return self

    def list_all_regions(self):
        region_list = [self]
        for r in self.regions:
            region_list.extend(r.list_all_regions())

        return region_list

    def enter_world(self, n, idx=None, arriving_from=None):
        boo_idx = self.population.find_indexes(idx)
        self.population.gravity[boo_idx] = np.zeros((n, 2))

        return super().enter_world(n, idx, arriving_from)
