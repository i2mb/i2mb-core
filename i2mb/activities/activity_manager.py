from typing import Protocol

import numpy as np

from i2mb import Model
from i2mb.activities import ActivityProperties, ActivityDescriptorProperties, TypesOfLocationBlocking
from i2mb.activities.base_activity import ActivityList, ActivityController
from i2mb.engine.relocator import Relocator
from i2mb.utils import time


class TriggeredActivityHandler(Protocol):
    def register_handler_action(self, key, action):
        ...

    def execute_handler_actions(self):
        ...


class LocationTriggeredActivityHandler:
    def __init__(self, population):
        self.population = population
        self.__trigger_on_location = {}

    def register_handler_action(self, location_ix, action):
        self.__trigger_on_location.setdefault(location_ix, []).append(action)

    def execute_handler_actions(self):
        # TODO: Maybe computing the difference between population.regions and registered location triggers
        #  and then looping over that difference might be faster than current implementation.
        seen_parents = set()
        for region in self.population.regions:
            if region.parent is not None:
                seen_parents.add(region.parent)

            if region.index in self.__trigger_on_location:
                for action in self.__trigger_on_location[region.index]:
                    action(region)

        for parent in seen_parents:
            if parent.index in self.__trigger_on_location:
                for action in self.__trigger_on_location[parent.index]:
                    action(parent)


class ActivityManager(Model):
    file_headers = ["id", "activity", "start", "duration", "location"]

    def __init__(self, population, relocator: 'Relocator' = None, write_diary=False):
        super().__init__()

        self.write_diary = write_diary
        self.relocator = None
        self.region_index = np.array([-1, -1, -1])
        self.blocked_locations = np.zeros(len(self.region_index), dtype=bool)

        population_size = len(population)
        self.population = population
        self.activity_list = ActivityList(population)

        self.current_activity = np.full(population_size, -1, dtype=int)
        self.current_descriptors = np.full((population_size, len(ActivityDescriptorProperties)), -1, dtype=int)

        # Activity Diary
        self.file = None

        # Activity rankings
        self.activity_types = np.array([])
        self.activity_ranking = {}

        # TriggerHandlers
        self.location_activity_handler = LocationTriggeredActivityHandler(self.population)

        self.link_relocator(relocator)

    def post_init(self, base_file_name=None):
        self.register_location_activities()
        self.activity_types = np.array([act_type.__name__ for act_type in self.activity_list.activity_types])

        # Activity Diary
        if self.write_diary:
            self.register_activity_stop_logger()
            super().post_init(base_file_name=base_file_name)
            self.base_file_name = f"{self.base_file_name}_activity_history.csv"
            self.file = open(self.base_file_name, "w+")
            self.file.write(",".join(self.file_headers) + "\n")

    def register_activity_stop_logger(self):
        for activity in self.activity_list.activities:
            activity.register_stop_callbacks(self.log_finished_activities)

    def log_finished_activities(self, activity_id, t, ids):
        if self.file is None:
            return

        if len(ids) == 0:
            return

        ids = self.population.index[ids]
        activity = self.activity_list.activities[activity_id]
        elapsed = activity.get_elapsed()[ids]
        start = activity.get_start()[ids]
        activity_type = [type(activity).__name__] * len(ids)
        location_ids = activity.get_location()[ids]
        locations = [type(loc_).__name__ for loc_ in self.region_index[location_ids, 2]]
        activity_log = np.vstack([ids, activity_type, start, elapsed, locations]).T
        self.file.write("\n".join([",".join(r) for r in activity_log]) + "\n")

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def generate_activity_ranking(self):
        for ix, act in enumerate(self.activity_list.activities):
            self.activity_ranking.setdefault(act.rank, list()).append((ix, act))

    def link_relocator(self, relocator):
        if relocator is not None:
            self.relocator = relocator
            self.region_index = relocator.universe.region_index
            self.blocked_locations = relocator.universe.blocked_locations

    def register_activity(self, activity):
        self.activity_list.add(activity)

    def pre_step(self, t):
        self.update_blocked_activities()
        self.stop_activities_with_duration(t)

    def step(self, t):

        if self.relocator is not None:
            self.location_activity_handler.execute_handler_actions()

        self.start_staged_activities()

    def post_step(self, t):
        self.update_current_activity()

    def start_staged_activities(self):
        """Do to blocked locations, or wait blocking requests, activity descriptors might be left in the stage area.
        Here, we try to start them again."""
        staged = self.current_descriptors[:, ActivityDescriptorProperties.act_idx] != -1
        self.start_activities(staged)

    def unblock_empty_location(self, region):
        self.__unblock_location(region)

    def __unblock_location(self, region):
        """If the parent region is blocked, we assume the at least one child blocked it. Therefore, we recursively
        check the parent to ensure all blocked children are considered before unlocking the parent. """
        # The children at listed starting in the second entry.
        blocked_children = region.blocked_locations[2:]
        if ~blocked_children.any():
            region.blocked = False

            if region.parent is None:
                return

            if region.parent.blocked:
                self.__unblock_location(region.parent)

    def unblock_locations(self, ids):
        if len(ids) == 0:
            return

        if not self.blocked_locations.any():
            return

        if not hasattr(self.population, "location"):
            return

        regions = np.array(list(set(self.population.location[ids])))
        reg_population = np.array([len(r.population) for r in regions])

        # We assume that only when the region has a single individual, stopping an activity unblocks the region
        regions = regions[reg_population <= 1]

        # Vectorized form
        np.frompyfunc(self.__unblock_location, 1, 0)(regions)

    def block_locations(self, ids, activity_descriptors):

        block_location = (activity_descriptors[:, ActivityDescriptorProperties.blocks_location] ==
                          TypesOfLocationBlocking.shared)

        # Parent blocking is always shared.
        block_parent_location = (activity_descriptors[:, ActivityDescriptorProperties.blocks_parent_location] >
                                 TypesOfLocationBlocking.no_blocking)

        # If the parent is blocked, ensure the location is also blocked
        block_location |= block_parent_location

        locations = activity_descriptors[block_location, ActivityDescriptorProperties.location_ix]
        block_parent_locations = activity_descriptors[block_parent_location, ActivityDescriptorProperties.location_ix]
        region_index = self.region_index
        if len(locations) > 0:
            # locations_selector = (region_index[:, [0]] == locations).any(axis=1)  # type: ignore
            self.blocked_locations[locations] = True

        if len(block_parent_locations) > 0:
            parent_ids = np.unique(region_index[block_parent_locations, 1])

            # Block the parent
            self.blocked_locations[parent_ids] = True

    def update_blocked_activities(self):
        blocked_for_ix = ActivityProperties.blocked_for
        in_progress_ix = ActivityProperties.in_progress
        mask = self.activity_list.activity_values[:, blocked_for_ix, :] > 0
        mask &= self.activity_list.activity_values[:, in_progress_ix, :] == 0
        if mask.any():
            self.activity_list.activity_values[:, blocked_for_ix, :][mask] -= 1

    def stop_activities_with_duration(self, t):
        """Activities that have duration set to a number greater than 0 are in_progress for that length of time. Once
        the elapsed time equals the duration time, the activity is stopped."""
        ids = self.current_activity != -1
        activities = self.current_activity[ids]
        ids = self.population.index[ids]
        elapsed = self.activity_list.get_elapsed(ids, activities)
        duration = self.activity_list.get_duration(ids, activities)

        # We could check for in_progress explicitly. But it adds one more memory access and vector calculation. So,
        # we assume consistency is maintained in start_activities, and stop_activities.
        stop_ids = ids[(elapsed == duration) & (duration > 0)]
        self.stop_activities(t, stop_ids)

    def update_current_activity(self):
        """This method supports updating multiple activities for a single agent simultaneously."""
        current_activities = self.activity_list.get_in_progress(slice(None), slice(None)) == 1
        self.activity_list.get_elapsed(slice(None), slice(None))[current_activities] += 1
        self.activity_list.get_accumulated(slice(None), slice(None))[current_activities] += 1

    def stage_activity(self, act_descriptors: np.ndarray, ids):
        if not (isinstance(ids, np.ndarray) and isinstance(ids.dtype, int)):
            ids = self.population.index[ids]

        if len(act_descriptors) == 1 and len(ids) > 1:
            act_descriptors = np.tile(act_descriptors, (len(ids), 1))

        activities = act_descriptors[:, ActivityDescriptorProperties.act_idx]
        non_blocked_activities = self.activity_list.get_blocked_for(ids, activities) == 0
        act_descriptors = act_descriptors[non_blocked_activities, :]
        ids = ids[non_blocked_activities]

        self.current_descriptors[ids] = act_descriptors

        return non_blocked_activities

    def start_activities(self, ids):
        ids = self.population.index[ids]
        if len(ids) == 0:
            return

        act_descriptors = self.current_descriptors[ids, :]
        if (act_descriptors[:, ActivityDescriptorProperties.act_idx] == -1).any():
            un_staged = ids[act_descriptors[:, ActivityDescriptorProperties.act_idx] == -1]
            raise ValueError(f"Starting un-staged activity for the following ids: {un_staged}.")

        # Check that locations are empty for the wait blocking type
        ids = self.check_for_wait_blockings(ids, act_descriptors)
        if len(ids) == 0:
            return

        # Relocate agents to trigger space dependent adjustments.
        location_ids = act_descriptors[:, ActivityDescriptorProperties.location_ix]
        ids = self.relocate_agents(ids, location_ids)
        if len(ids) == 0:
            return

        # Refresh descriptors to correspond to moved ids
        act_descriptors = self.current_descriptors[ids, :]
        activities = act_descriptors[:, ActivityDescriptorProperties.act_idx]

        self.current_activity[ids] = activities
        self.activity_list.set_start(ids, activities, time())
        self.activity_list.set_in_progress(ids, activities, 1,)
        self.activity_list.set_duration(ids, activities,
                                        act_descriptors[:, ActivityDescriptorProperties.duration])
        self.activity_list.set_blocked_for(ids, activities,
                                           act_descriptors[:, ActivityDescriptorProperties.block_for])
        self.activity_list.set_location(ids, activities,
                                        act_descriptors[:, ActivityDescriptorProperties.location_ix])

        self.current_descriptors[ids, :] = -1

        self.block_locations(ids, act_descriptors)

        for act in self.activity_list.activities:
            start_activity = ids[(self.current_activity[ids] == act.id)]
            if len(start_activity) > 0:
                act.start_activity(time(), start_activity)
                act.finalize_start(start_activity)

    def relocate_agents(self, ids, location_ix):
        if self.relocator is None:
            # warning("No relocator registered, Ignoring location changes.", RuntimeWarning, True)
            return ids

        region_index = self.relocator.universe.region_index
        update_current = location_ix != 0
        relocated_ids = np.ones_like(ids, dtype=bool)
        same_location = (self.population.location[ids] == region_index[location_ix, 2]).ravel()
        if update_current.any():
            relocated_ids = np.zeros_like(ids, dtype=bool)
            ids = self.population.index[ids]
            locations = region_index[location_ix, 2]
            unique_locations = set(locations).difference({-1})

            for loc in unique_locations:
                moving_ids = ids[locations == loc]
                moved_ids = self.relocator.move_agents(moving_ids, loc)
                ids_selector = (ids.reshape(-1, 1) == moved_ids).any(axis=1)
                relocated_ids[ids_selector] = True

        return ids[relocated_ids | ~update_current | same_location]

    def reset_current_activity(self, ids=None):
        if ids is None:
            ids = slice(None)

        ids = self.population.index[ids]
        ids = ids[(self.current_activity[ids] != -1)]
        if len(ids) == 0:
            return

        activities = self.current_activity[ids]
        self.activity_list.set_in_progress(ids,  activities, 0)
        self.activity_list.set_start(ids,  activities, 0)
        self.activity_list.set_duration(ids,  activities, 0)
        self.activity_list.set_elapsed(ids,  activities, 0)
        # self.set_current_activity_property(ActivityProperties.blocked_for, 0, ids)
        self.activity_list.set_location(ids,  activities, 0)

    def stop_activities_on_relocation(self, ids, old_location):
        self.stop_activities(time(), ids)

    def stop_activities(self, t, stop_ids):
        if len(stop_ids) == 0:
            return

        stop_ids = self.population.index[stop_ids]

        for act in self.activity_list.activities:
            stop_ids_ = stop_ids[(self.current_activity == act.id)[stop_ids]]
            if len(stop_ids_) > 0:
                act.stop_activity(t, stop_ids_)

        self.reset_current_activity(stop_ids)
        self.unblock_locations(stop_ids)
        self.current_activity[stop_ids] = -1

    def check_for_wait_blockings(self, ids, act_descriptors):
        if self.relocator is None:
            return ids

        locations_ixs = act_descriptors[:, ActivityDescriptorProperties.location_ix].copy()
        locations_ixs[locations_ixs == 0] = [r.index for r in self.population.location[ids][locations_ixs == 0]]
        location_occupation = np.array([r.population is not None and len(r.population) or 0
                                        for r in self.region_index[locations_ixs, 2]]) < 1
        wait_blocking = act_descriptors[:, ActivityDescriptorProperties.blocks_location] == TypesOfLocationBlocking.wait
        can_block_do_to_wait = location_occupation & wait_blocking

        return ids[~wait_blocking | can_block_do_to_wait]

    def register_location_activities(self):
        if self.relocator is not None:
            # Register event handling functions
            self.relocator.register_on_region_exit_action(self.stop_activities_on_relocation)
            self.relocator.register_on_region_empty_action(self.unblock_empty_location)

            for r in self.relocator.universe.list_all_regions():
                if not hasattr(r, "local_activities"):
                    continue

                for act in r.local_activities:
                    act_type = act.activity_class
                    if act_type not in self.activity_list.activity_types:
                        activity = act_type(self.population)
                        self.register_activity(activity)

