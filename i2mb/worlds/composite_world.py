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

from .world_base import World
from ..engine.agents import AgentList
from ..utils import cache_manager


class CompositeWorld(World):
    def __init__(self, dims=None, population: AgentList = None, regions=None, origin=None, map_file=None,
                 containment=False, waiting_room=False, rotation=0, scale=1):
        self.regions = []
        super().__init__(dims, origin=origin, rotation=rotation, scale=scale, subareas=self.regions)

        self.waiting_room = waiting_room
        self.containment = containment
        self.region_origins = np.array([])
        self.region_dimensions = np.array([])

        if regions:
            self.add_regions(regions)

        if map_file is not None:
            self.load_map(map_file)

        if population is None:
            self.population = None
            self.location = np.array([])
            self.position = np.array([])
            self.gravity = np.array([])
            self.containment_region = np.array([])
            self.remain = np.array([])
            if waiting_room:
                self.in_waiting_room = np.array([])

            return

        self.population = population
        n = len(population)
        self.__in_region = np.zeros(n, dtype=bool)
        self.location = np.array([self] * len(population))
        self.position = self.enter_world(n)
        self.containment_region = np.empty((n,), dtype=object)
        self.home = np.empty((n,), dtype=object)
        self.remain = np.zeros((n,), dtype=bool)
        population.add_property("location", self.location)
        population.add_property("remain", self.remain)
        population.add_property("position", self.position)
        population.add_property("containment_region", self.containment_region)
        population.add_property("home", self.home)
        population.add_property("regions", {self}, l_property=True)
        self.gravity = np.zeros((n, 2))
        population.add_property("gravity", self.gravity)

        if waiting_room:
            self.in_waiting_room = np.zeros((n, 1), dtype=bool)
            population.add_property("in_waiting_room", self.in_waiting_room)

    def __hash__(self):
        return hash(id(self))

    def check_positions(self, mask):
        if hasattr(self.population, "regions"):
            for r in self.population.regions:
                r_mask = mask[r.population.index]
                r.check_positions(r_mask)

        if self.is_empty():
            return

        positions = self.population.position[self.location == self]
        if len(positions) == 0:
            return

        mask = mask[self.location == self]
        check_top = mask.ravel() & (positions[:, 0] > self.dims[0])
        check_right = mask.ravel() & (positions[:, 1] > self.dims[1])
        check_left = mask.ravel() & (positions[:, 0] < 0)
        check_bottom = mask.ravel() & (positions[:, 1] < 0)

        positions[check_top, 0] = self.dims[0]
        positions[check_right, 1] = self.dims[1]
        positions[check_left, 0] = 0
        positions[check_bottom, 1] = 0
        self.population.position[self.location == self] = positions
        for lm in self.landmarks:
            lm.remove_overlap()

    def get_containment_regions(self, idx):
        return self.containment_region[idx]

    def get_home_regions(self, idx):
        return self.home[idx]

    def load_map(self, map_file):
        pass

    def move_agents(self, idx, region):
        """Only the outer most region can move agents between two contained worlds"""
        idx = self.population.index[idx][~self.population.remain[idx]]
        if len(idx) == 0:
            return

        departed_from_regions = self.depart_current_region(idx)
        self.enter_region(idx, region, departed_from_regions)
        cache_manager.invalidate()

    def enter_region(self, idx, region, departed_from_regions):
        idx_ = idx
        region = region.get_entrance_sub_region()
        if region.population is not None:
            old_idx = region.population.index
            idx = np.union1d(old_idx, idx)

        region.population = self.population[idx]
        region.position = self.position[idx]
        self.location[idx] = region
        self.position[idx_] = region.enter_world(len(idx_), idx=idx_, arriving_from=departed_from_regions)
        self.gravity[idx_] = np.zeros((len(idx_), 2))
        region.location = self.location[idx]
        self.population.regions.add(region)
        if self not in self.location and self in self.population.regions:
            self.population.regions.remove(self)

    def depart_current_region(self, idx):
        depart = set(self.location[idx]) - {self}
        for r in depart:
            old_idx = r.population.index
            new_idx = np.setdiff1d(old_idx, idx)
            leaving_idx = np.intersect1d(old_idx, idx)
            r.exit_world(leaving_idx)
            r.population = self.population[new_idx]
            r.location = self.location[new_idx]
            r.position = self.position[new_idx]
            if r.is_empty():
                self.population.regions.remove(r)

        return self.location[idx]

    def is_empty(self):
        if hasattr(self, "population") and len(self.population) > 0:
            return False

        return True

    def add_regions(self, regions):
        self.regions.extend(regions)
        for region in regions:
            region.parent = self

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

    def get_absolute_origin(self):
        if self.parent is None:
            return self.origin

        return self.origin + self.parent.get_absolute_origin()

    def get_absolute_positions(self):
        abs_pos = np.zeros((len(self.population), 2))
        seen_index = np.array([])
        for r in self.population.regions:
            idx = r.population.index
            seen_index = np.union1d(seen_index, idx)
            abs_pos[idx] = r.get_absolute_origin() + r.population.position

        idx_ = np.setdiff1d(self.population.index, seen_index)
        abs_pos[idx_] = self.population.position[idx_] + self.origin

        return abs_pos

    def draw_world(self, ax=None, origin=(0, 0), **kwargs):
        bbox = kwargs.get("bbox", False)
        self._draw_world(ax, origin=self.origin + origin, **kwargs)
        for region in self.regions:
            region.draw_world(ax=ax, bbox=bbox, origin=self.origin + origin)

    def _draw_world(self, ax, bbox=False, origin=(0, 0), **kwargs):
        ax.add_patch(Rectangle(origin, *self.dims, fill=False, linewidth=1.2, edgecolor='gray'))

    def get_entrance_sub_region(self):
        return self
