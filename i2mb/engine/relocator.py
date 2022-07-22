import numpy as np

from i2mb.utils import cache_manager

from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    from i2mb.engine.agents import AgentList
    from i2mb.worlds import CompositeWorld


class Relocator:
    """Relocator is the class in charge of moving agents form one location to another."""

    def __init__(self, population: 'AgentList', universe: 'CompositeWorld'):
        self.universe = universe
        self.population = population

        n = len(population)
        self.location = np.array([universe] * n)  # type: np.ndarray[CompositeWorld]
        self.position = np.full((n, 2), -0.)

        self.visit_counter = {}

        population.add_property("location", self.location)
        population.add_property("position", self.position)

        # Events call backs
        self.on_region_exit_actions = []
        self.on_region_enter_actions = []
        self.on_region_empty_actions = []
        self.on_move_cancelled_actions = []

    def execute_on_region_exit_actions(self, idx, region):
        for action in self.on_region_exit_actions:
            action(idx, region)

    def execute_on_region_enter_actions(self, idx, region, departed_from_regions):
        for action in self.on_region_enter_actions:
            action(idx, region, departed_from_regions)

    def execute_on_region_empty_actions(self, region):
        for action in self.on_region_empty_actions:
            action(region)

    def execute_on_move_cancelled_actions(self, ids, region):
        for action in self.on_move_cancelled_actions:
            action(ids, region)

    def register_on_region_exit_action(self,
                                       func: Union[Callable[[np.ndarray, 'CompositeWorld'], None],
                                                   list[Callable[[np.ndarray, 'CompositeWorld'], None]]]):
        if isinstance(func, Callable):
            self.on_region_exit_actions.append(func)
        else:
            self.on_region_exit_actions.extend(func)

    def register_on_region_enter_action(self,
                                       func: Union[Callable[[np.ndarray, 'CompositeWorld'], None],
                                                   list[Callable[[np.ndarray, 'CompositeWorld'], None]]]):
        if isinstance(func, Callable):
            self.on_region_enter_actions.append(func)
        else:
            self.on_region_enter_actions.extend(func)

    def register_on_region_empty_action(self,
                                        func: Union[Callable[['CompositeWorld'], None],
                                                    list[Callable[['CompositeWorld'], None]]]):
        if isinstance(func, Callable):
            self.on_region_empty_actions.append(func)
        else:
            self.on_region_empty_actions.extend(func)

    def register_on_move_cancelled_action(self,
                                          func: Union[Callable[[np.ndarray, 'CompositeWorld'], None],
                                                      list[Callable[['CompositeWorld'], None]]]):
        if isinstance(func, Callable):
            self.on_move_cancelled_actions.append(func)
        else:
            self.on_move_cancelled_actions.extend(func)

    def move_agents(self, idx, region):
        idx = self.population.index[idx]

        # Remove ids that are already in region
        mask = self.population.location[idx] == region
        idx = self.population.index[idx][~mask]
        if len(idx) == 0:
            return idx

        departed_ix = self.depart_current_region(idx, region)
        cancelled_ix = idx[~departed_ix]
        self.execute_on_move_cancelled_actions(cancelled_ix, region)
        idx = idx[departed_ix]
        departed_from_regions = self.location[idx]
        self.enter_region(idx, region, departed_from_regions)
        cache_manager.invalidate()
        return idx

    def depart_current_region(self, idx, destination):
        depart = set(self.location[idx]) - {self}
        can_enter_destination = np.ones_like(idx, dtype=bool)
        for r in depart:  # type: CompositeWorld
            old_idx = r.population.index
            new_idx = np.setdiff1d(old_idx, idx)
            leaving_idx = np.intersect1d(old_idx, idx)
            if self.check_entry_route_locked(r, destination):
                cancelled = (idx.reshape(-1, 1) == leaving_idx).any(axis=1)
                can_enter_destination[cancelled] = False
                continue

            self.execute_on_region_exit_actions(leaving_idx, r)
            self.execute_transfer_route(destination, leaving_idx, r)
            r.population = self.population[new_idx]
            r.location = self.location[new_idx]
            r.position = self.position[new_idx]
            if r.is_empty():
                self.population.regions.remove(r)
                self.execute_on_region_empty_actions(r)

        return can_enter_destination

    @staticmethod
    def check_entry_route_locked(origin: 'CompositeWorld', destination: 'CompositeWorld'):
        entry_matrix = origin.entry_route.reshape(-1, 1) == destination.entry_route
        entry_levels = ~(entry_matrix.any(axis=0))
        blocked = [r.blocked for r in destination.entry_route[entry_levels]]
        if np.any(blocked):
            return True
        else:
            return False

    def execute_transfer_route(self, destination, leaving_idx, origin: 'CompositeWorld'):
        entry_matrix = origin.entry_route.reshape(-1, 1) == destination.entry_route
        entry_levels = ~(entry_matrix.any(axis=0))
        exit_levels = ~(entry_matrix.any(axis=1))
        for level in origin.entry_route[exit_levels]:
            level.active_exit_world(leaving_idx, self.population)

        for level in destination.entry_route[entry_levels][:-1]:
            level.active_enter_world(len(leaving_idx), leaving_idx, [origin])

    def enter_region(self, idx, region, departed_from_regions):
        if len(idx) == 0:
            return

        idx_ = idx
        region.prepare_entrance(idx, self.population)
        entrance_region = region.get_entrance_sub_region()
        self.update_visit_counter(idx, entrance_region, region)

        region = entrance_region
        if region.population is not None:
            old_idx = region.population.index
            idx = np.union1d(old_idx, idx)

        region.population = self.population[idx]
        region.position = self.position[idx]
        self.location[idx] = region
        self.position[idx_] = region.active_enter_world(len(idx_), idx=idx_, arriving_from=departed_from_regions)

        region.location = self.location[idx]
        self.population.regions.add(region)
        if self not in self.location and self in self.population.regions:
            self.population.regions.remove(self)

        self.execute_on_region_enter_actions(idx, region, departed_from_regions)

    def update_visit_counter(self, idx, entrance_region, region):
        if type(region) not in self.visit_counter:
            # avoids allocation of default value when the key already exists.
            self.visit_counter[type(region)] = np.zeros(len(self.population), dtype=int)

        if type(entrance_region) not in self.visit_counter:
            # avoids allocation of default value when the key already exists.
            self.visit_counter[type(entrance_region)] = np.zeros(len(self.population), dtype=int)

        self.visit_counter[type(region)][idx] += 1
        if entrance_region is not region:
            self.visit_counter[type(entrance_region)][idx] += 1

    def get_absolute_positions(self):
        abs_pos = np.zeros((len(self.population), 2))
        seen_index = np.array([])
        for r in self.population.regions:
            idx = r.population.index
            seen_index = np.union1d(seen_index, idx)
            abs_pos[idx] = r.get_absolute_origin() + r.population.position

        idx_ = np.setdiff1d(self.population.index, seen_index)
        abs_pos[idx_] = self.population.position[idx_] + self.universe.origin

        return abs_pos
