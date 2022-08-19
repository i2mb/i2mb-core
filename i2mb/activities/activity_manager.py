import numpy as np

from i2mb import Model
from i2mb.activities import ActivityProperties, ActivityDescriptorProperties, TypesOfLocationBlocking
from i2mb.activities.base_activity import ActivityList, ActivityController
from i2mb.engine.relocator import Relocator
from i2mb.utils import time


class ActivityManager(Model):
    file_headers = ["id", "activity", "start", "duration", "location"]

    def __init__(self, population, relocator: 'Relocator' = None, write_diary=False):
        super().__init__()

        self.controllers = []
        self.__controllers = []
        self.activity_controllers = {}

        self.write_diary = write_diary
        self.relocator = None
        self.region_index = np.array([-1, -1, -1])
        self.blocked_locations = np.zeros(len(self.region_index), dtype=bool)

        population_size = len(population)
        self.population = population
        self.activity_list = ActivityList(population)

        self.current_activity = np.full(population_size, -1, dtype=int)
        self.current_descriptors = np.full((population_size, len(ActivityDescriptorProperties)), -1, dtype=int)
        self.current_activity_interruptable = np.ones(len(self.population), dtype=bool)

        # Activity Diary
        self.file = None

        # Activity rankings
        self.activity_ranking = {}

        self.link_relocator(relocator)

    def post_init(self, base_file_name=None):
        self.register_location_activities()
        self.register_activity_on_stop_method()
        self.__execute_controller_registration()

        # Activity Diary
        if self.write_diary:
            self.register_activity_stop_logger()
            super().post_init(base_file_name=base_file_name)
            self.base_file_name = f"{self.base_file_name}_activity_history.csv"
            self.file = open(self.base_file_name, "w+")
            self.file.write(",".join(self.file_headers) + "\n")

    def register_activity_on_stop_method(self):
        for activity in self.activity_list.activities:
            activity.register_stop_callbacks(self.update_interruptable_flag)

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

    def update_interruptable_flag(self, act_id, t, stop_selector):
        self.current_activity_interruptable[stop_selector] = True

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
        return self.activity_list.activities[activity.id]

    def pre_step(self, t):
        self.update_blocked_activities()
        self.stop_activities_with_duration(t)

    def step(self, t):
        # Initialize with inactive population
        inactive = self.current_activity == -1
        new_activities = self.current_activity == -1
        for controller in self.controllers:  # controller: ActivityController
            new_activities |= controller.has_new_activity(inactive)

        new_activities &= self.collect_new_controller_descriptors(new_activities)
        self.start_activities(self.population.index[new_activities])

    def post_step(self, t):
        self.update_current_activity()

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
            unblock_activities = self.activity_list.activity_values[:, blocked_for_ix, :] == 0
            unblock_activities &= mask
            if unblock_activities.any():
                for act_ix in range(unblock_activities.shape[-1]):
                    activity = self.activity_list.activities[act_ix]
                    activity_unblock = unblock_activities[:, act_ix].ravel()
                    if activity_unblock.any():
                        activity.run_unblock_callbacks(time(), activity_unblock)

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

        if len(act_descriptors) != len(ids):
            raise RuntimeError(f"Something is not right length of ids and descriptors does not match"
                               f" {act_descriptors}, {ids}")

        # Check activities are not blocked
        activities = act_descriptors[:, ActivityDescriptorProperties.act_idx]
        non_blocked_activities = self.activity_list.get_blocked_for(ids, activities) == 0

        # Check if the current activity is uninterruptible
        interruptable = self.current_activity_interruptable[ids]

        # Only stage activities that are not blocked and that where the current activity can be interrupted
        staging = non_blocked_activities & interruptable

        # Check that locations are empty for the wait blocking type
        staging &= self.check_for_wait_blockings(ids, act_descriptors)

        # Relocate agents to trigger space dependent adjustments.
        location_ids = act_descriptors[:, ActivityDescriptorProperties.location_ix]
        staging &= self.relocate_agents(ids, location_ids, staging)

        act_descriptors = act_descriptors[staging, :]
        ids = ids[staging]
        self.current_descriptors[ids] = act_descriptors[:, ]

        return staging

    def start_activities(self, ids):
        if len(ids) == 0:
            return

        act_descriptors = self.current_descriptors[ids]
        activities = act_descriptors[:, ActivityDescriptorProperties.act_idx]

        # Remove cases where staged activity is the same as current activity.
        current_activities = self.current_activity[ids]
        new_activity = current_activities != activities

        # Removed unused descriptors
        self.current_descriptors[ids[~new_activity], :] = -1
        act_descriptors = act_descriptors[new_activity, :]
        activities = activities[new_activity]
        ids = ids[new_activity]

        if len(ids) == 0:
            return

        self.stop_activities(time(), ids)
        self.__start_activities(act_descriptors, activities, ids)

    def collect_new_controller_descriptors(self, ids_selector):
        ids = self.population.index[ids_selector]
        staged = np.zeros_like(ids, dtype=bool)
        stage_idx = np.arange(len(staged))
        full_staged = np.zeros(len(self.population), dtype=bool)
        for controller in self.controllers:  # type: ActivityController
            not_staged = ~staged
            act_descriptors, new_ids_selector = controller.get_new_activity(ids[not_staged])
            if ~new_ids_selector.any():
                continue

            staged_ = self.stage_activity(act_descriptors, ids[not_staged][new_ids_selector])

            full_staged[ids[not_staged][new_ids_selector]] = staged_
            staged[stage_idx[not_staged][new_ids_selector]] = staged_

            if staged.all():
                break

        return full_staged

    def __start_activities(self, act_descriptors, activities, ids):
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
        self.current_activity_interruptable[ids] = act_descriptors[:, ActivityDescriptorProperties.interruptable]
        self.block_locations(ids, act_descriptors)

        for act in self.activity_list.activities:
            start_activity = ids[(self.current_activity[ids] == act.id)]
            if len(start_activity) > 0:
                act.start_activity(time(), start_activity)
                act.finalize_start(start_activity)

    def relocate_agents(self, ids, location_ix, ids_selector):
        if self.relocator is None:
            # warning("No relocator registered, Ignoring location changes.", RuntimeWarning, True)
            return np.ones_like(ids, dtype=bool)

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
                moving_ids = ids[(locations == loc) & ids_selector]
                moved_ids = self.relocator.move_agents(moving_ids, loc)
                ids_selector_ = (ids.reshape(-1, 1) == moved_ids).any(axis=1)
                relocated_ids[ids_selector_] = True

        return relocated_ids | ~update_current | same_location

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
            return np.ones_like(ids, dtype=bool)

        locations_ixs = act_descriptors[:, ActivityDescriptorProperties.location_ix].copy()
        locations_ixs[locations_ixs == 0] = [r.index for r in self.population.location[ids][locations_ixs == 0]]
        location_empty = np.array([r.population is not None and len(r.population) or 0
                                        for r in self.region_index[locations_ixs, 2]]) < 1
        wait_blocking = act_descriptors[:, ActivityDescriptorProperties.blocks_location] == TypesOfLocationBlocking.wait
        can_block_do_to_wait = location_empty & wait_blocking

        return ~wait_blocking | can_block_do_to_wait

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

    def register_activity_controller(self, controller: 'ActivityController', activity=None, z_order=None):
        self.__controllers.append((controller, controller.registration_callback, activity))
        self.controllers.append(controller)
        if z_order is None:
            controller.z_order = len(self.__controllers)

        else:
            if z_order < 0:
                z_order = len(self.__controllers) - z_order

            controller.z_order = z_order

        self.controllers = sorted(self.controllers, key=lambda x: x.z_order)

    def __execute_controller_registration(self):
        for controller, registration_callback, activity in self.__controllers:
            # Register a global activity controller
            if activity is not None:
                self.activity_controllers[activity.id] = controller
                return

            registration_callback(self, self.relocator.universe)
