import numpy as np
from matplotlib.patches import Rectangle

from .world_base import World
from ..engine.particle import ParticleList
from ..utils import cache_manager


class CompositeWorld(World):
    def __init__(self, dims=None, population: ParticleList = None, regions=None, origin=None, map_file=None,
                 containment=False, waiting_room=False):
        super().__init__()
        self.dims = dims is not None and dims or (1, 1)
        self.origin = origin
        self.waiting_room = waiting_room
        self.containment = containment
        self.regions = []
        self.region_origins = np.array([])
        self.region_dimensions = np.array([])

        if regions:
            self.add_regions(regions)

        if map_file is not None:
            self.load_map(map_file)

        if population is None:
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
        for r in self.regions:
            if r.is_empty():
                continue

            r_mask = mask[r.population.index]
            r.check_positions(r_mask)

        if self.is_empty():
            return

        positions = self.population.position[self.location == self]
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

    def move_particles(self, idx, region):
        from . import House
        idx = self.population.index[idx][~self.population.remain[idx]]

        if isinstance(region, House):
            if not (self.population.home[idx] == region).all():
                raise RuntimeError("Moving someone to the wrong house")

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

        idx_ = idx
        if hasattr(region, 'population'):
            old_idx = region.population.index
            idx = np.union1d(old_idx, idx)

        region.population = self.population[idx]
        region.position = self.position[idx]
        self.location[idx] = region
        self.position[idx_] = region.enter_world(len(idx_), idx=idx_)
        self.gravity[idx_] = np.zeros((len(idx_), 2))

        region.location = self.location[idx]
        self.population.regions.add(region)
        if self not in self.location and self in self.population.regions:
            self.population.regions.remove(self)

        cache_manager.invalidate()

    def is_empty(self):
        if hasattr(self, "population") and len(self.population) > 0:
            return False

        return True

    def add_regions(self, regions):
        self.regions.extend(regions)
        region_origins = list(self.region_origins)
        region_dimensions = list(self.region_dimensions)
        region_origins.extend([r.origin for r in regions])
        region_dimensions.extend([r.dims for r in regions])

        self.region_origins = np.array(region_origins)
        self.region_dimensions = np.array(region_dimensions)

        region_origin = np.min(self.region_origins, axis=0)
        for d, v in enumerate(self.origin):
            if v > region_origin[d]:
                self.origin[d] = region_origin[d]

        region_dimensions = np.max(self.region_origins + self.region_dimensions, axis=0)
        for d, v in enumerate(self.dims):
            if v < region_dimensions[d]:
                self.dims[d] = region_dimensions[d] - self.origin[d]

    def get_absolute_positions(self):
        abs_pos = np.zeros((len(self.population), 2))
        seen_index = np.array([])
        for r in self.regions:
            if r.is_empty():
                continue

            idx = r.population.index
            seen_index = np.union1d(seen_index, idx)
            abs_pos[idx] = r.population.position + r.origin

        idx_ = np.setdiff1d(self.population.index, seen_index)
        abs_pos[idx_] = self.population.position[idx_] + self.origin

        return abs_pos

    def draw_world(self, ax=None, **kwargs):
        bbox = kwargs.get("bbox", False)
        for region in self.regions:
            region.draw_world(ax=ax, bbox=bbox)

        self._draw_world(ax, **kwargs)

    def _draw_world(self, ax, bbox=False):
        ax.add_patch(Rectangle(self.origin, *self.dims, fill=False, linewidth=1.2, edgecolor='gray'))
