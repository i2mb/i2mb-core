from typing import Union
from warnings import warn

import numpy as np

from i2mb import Model
from i2mb.activities.base_activity import ActivityList, ActivityNone
from i2mb.activities.base_activity_descriptor import ActivityDescriptorQueue, create_null_descriptor_for_act_id, \
    convert_activities_to_descriptors


def enforce_unique_resource_utilization(location_mask, next_activity_location):
    # Compute the index on the self.locations_ids vector
    pos_values = np.hstack([np.argwhere(location_mask.any(axis=1)),
                            np.take(next_activity_location, np.argwhere(location_mask.any(axis=1)))])
    pos_values_ordered = pos_values[pos_values[:, 1].argsort(), :]
    people_idx = pos_values_ordered[1:, 0][np.diff(pos_values_ordered[:, 1]) > 0].astype(int)
    first_come = np.ones_like(next_activity_location, dtype=bool)
    first_come[pos_values_ordered[0, 0]] = False
    first_come[people_idx] = False
    return first_come


class ActivityManager(Model):
    def __init__(self, population, world=None, activities: Union[ActivityList, None] = None):

        if activities is None:
            activities = ActivityList(population)

        self.activities = activities
        self.current_activity = self.activities.current_activity
        self.world = world
        self.population = population

        # Location occupancy management
        self.location_blocked = np.array([])
        self.location_ids = np.array([])
        self.register_available_locations()
        self.current_location_id = np.zeros_like(self.activities.current_activity, dtype=int) - 1

        # Default activity can be changed during runtime
        self.current_default_activity = np.zeros_like(self.activities.current_activity)

        self.activities.set_current_activity_property(self.activities.in_progress_ix, 1)
        self.current_activity_rank = np.zeros_like(self.current_activity)

        # FIFO queue to hold planned activities
        self.planned_activities = ActivityDescriptorQueue(len(population), 15)

        # FIFO queue to respect postponed activities
        self.postponed_activities = ActivityDescriptorQueue(len(population), 15)

        # LIFO queue to create interruption chains
        self.interrupted_activities = ActivityDescriptorQueue(len(population), 15)

        # FIFO queue to hold triggered activities
        self.triggered_activities = ActivityDescriptorQueue(len(population), 15)

        self.starting_activity = np.zeros((len(self.population)), dtype=bool)

        self.activity_ranking = {}

        if self.world is not None:
            self.register_available_locations()

    def post_init(self, base_file_name=None):
        self.generate_activity_ranking()

    def step(self, t):
        self.starting_activity[:] = False
        self.unblock_empty_locations()
        self.update_current_activity()
        self.stop_activities_after_location_changed(t)
        self.stop_activities_with_duration(t)
        self.consume_interrupted_activities(t)
        self.consume_postponed_activities(t)
        self.consume_triggered_activities(t)
        self.consume_planned_activities(t)
        self.initiate_activities(t)

    def pause_activities(self, have_new_activities):
        # pause activities
        act_pause = have_new_activities.copy()
        act_pause &= self.activities.get_current_activity_property(self.activities.duration_ix) > 0
        # act_pause &= self.activities.get_current_activity_property(self.activities.blocked_for_ix) == 0
        act_pause &= self.current_activity != self.current_default_activity
        if act_pause.any():
            paused_descriptors = convert_activities_to_descriptors(
                self.current_activity[act_pause],
                self.activities.activity_values[act_pause, :, self.current_activity[act_pause]],
                self.current_location_id[act_pause],
                self.activities.current_descriptors[act_pause],
                )

            self.interrupted_activities[act_pause].push(paused_descriptors)

    def stop_activities_with_duration(self, t):
        """Activities that have duration set to a number greater than 0 are in_progress for that length of time. Once
        the elapsed time equals the duration time, the activity is stopped and the agent's default activity is
        resumed. There are no guaranties that the queue management will assign a new activity to the agent."""
        elapsed_time = self.activities.get_current_activity_property(self.activities.elapsed_ix)
        duration = self.activities.get_current_activity_property(self.activities.duration_ix)
        duration_mask = (duration > 0) & (duration == elapsed_time)
        stop_ids = self.population.index[duration_mask]
        if len(stop_ids) > 0:
            self.stop_activities(stop_ids, t)

    def stop_activities(self, stop_ids, t):
        self.unblock_locations(stop_ids)
        next_activity = create_null_descriptor_for_act_id(self.current_default_activity[stop_ids])
        self.stage_activities(stop_ids, next_activity, t)
        self.starting_activity[stop_ids] = True

    def update_current_activity(self):
        elapsed_ix = self.activities.elapsed_ix
        accumulated_ix = self.activities.accumulated_ix
        blocked_for_ix = self.activities.blocked_for_ix
        in_progress_ix = self.activities.in_progress_ix

        property_value = self.activities.get_current_activity_property(elapsed_ix)
        self.activities.set_current_activity_property(elapsed_ix, property_value + 1)

        property_value = self.activities.get_current_activity_property(accumulated_ix)
        self.activities.set_current_activity_property(accumulated_ix, property_value + 1)

        mask = self.activities.activity_values[:, blocked_for_ix, :] > 0
        mask &= self.activities.activity_values[:, in_progress_ix, :] == 0
        if mask.any():
            self.activities.activity_values[:, blocked_for_ix, :][mask] -= 1

    def relocate_agents(self, ids, location_ids):
        if len(self.location_ids) == 0:
            warn("No locations registered, make sure that self.register_available_locations is executed before "
                 "running the engine.", RuntimeWarning, True)

            self.current_location_id[ids] = location_ids
            return

        locations_selector = np.searchsorted(self.location_ids[:, 0],   location_ids)
        locations = self.location_ids[locations_selector, 2]
        unique_locations = set(locations).difference({-1})
        for loc in unique_locations:
            self.world.move_agents(ids[locations == loc], loc)

        self.current_location_id[ids] = location_ids

    def generate_activity_ranking(self):
        for ix, act in enumerate(self.activities.activities):
            self.activity_ranking.setdefault(act.rank, list()).append((ix, act))

    def consume_interrupted_activities(self, t):
        has_interrupted_activities = self.interrupted_activities.num_items > 0

        # We opt not to test for in_progress, as the default activity should be interrupted to finish any
        # interrupted activities.
        current_activity_is_finished = self.current_activity == self.current_default_activity
        can_resume_activity = has_interrupted_activities & current_activity_is_finished
        if can_resume_activity.any():
            resume_activities = self.interrupted_activities[can_resume_activity].pop()

            self.stage_activities(can_resume_activity, resume_activities, t)

    def consume_postponed_activities(self, t):
        have_postponed_activities = (self.postponed_activities.num_items > 0) & ~self.starting_activity
        if have_postponed_activities.any():
            resource_not_available = self.check_activity_resource_availability(have_postponed_activities,
                                                                               self.postponed_activities)

            ids = self.population.index[have_postponed_activities]
            have_postponed_activities[ids] &= ~resource_not_available
            if have_postponed_activities.any():
                self.pause_activities(have_postponed_activities)
                new_activities = self.postponed_activities[have_postponed_activities].pop()
                self.stage_activities(have_postponed_activities, new_activities, t)

    def consume_triggered_activities(self, t):
        have_triggered_activities = (self.triggered_activities.num_items > 0) & ~self.starting_activity
        if have_triggered_activities.any():
            resource_not_available = self.check_activity_resource_availability(have_triggered_activities,
                                                                               self.triggered_activities)

            ids = self.population.index[have_triggered_activities]
            have_triggered_activities[ids] &= ~resource_not_available
            if have_triggered_activities.any():
                self.pause_activities(have_triggered_activities)
                new_activities = self.triggered_activities[have_triggered_activities].pop()
                self.stage_activities(have_triggered_activities, new_activities, t)

    def consume_planned_activities(self, t):
        blocked_activities = np.array([], ndmin=2)
        starting_activities = self.starting_activity & (self.current_activity != self.current_default_activity)
        have_new_activities = (self.planned_activities.num_items > 0) & ~starting_activities
        if have_new_activities.any():
            self.remove_time_blocked_activities(have_new_activities)
            act_ready_to_start = (self.planned_activities.start <= t) & have_new_activities
            act_wait = self.activities.get_current_activity_property(self.activities.duration_ix) > 0
            act_wait &= self.activities.get_current_activity_property(self.activities.blocked_for_ix) > 0
            act_ready_to_start &= ~act_wait
            if act_ready_to_start.any():
                self.pause_activities(act_ready_to_start)
                not_postponed_act = self.postpone_activities(act_ready_to_start, self.planned_activities)
                act_ready_to_start &= not_postponed_act
                if act_ready_to_start.any():
                    new_activities = self.planned_activities[act_ready_to_start].pop()
                    self.stage_activities(act_ready_to_start, new_activities, t)

        if blocked_activities.shape != (1, 0):
            print(blocked_activities)

        return blocked_activities

    def stage_activities(self, has_activities_to_stage, activity_descriptors_to_stage, t):
        # Start times
        activity_descriptors_to_stage[:, 1] = t
        ids = self.population.index[has_activities_to_stage]
        applied_activities = self.activities.apply_descriptors(t, activity_descriptors_to_stage, ids)
        self.block_locations(activity_descriptors_to_stage)
        self.relocate_agents(ids, activity_descriptors_to_stage[:, 5])
        idx = ids[applied_activities]
        self.starting_activity[idx] = True

    def initiate_activities(self, t):
        if self.starting_activity.any():
            self.activities.start_activities(self.starting_activity)
            for act in self.activities.activities:
                start_activity = self.starting_activity & (self.current_activity == act.id)
                if start_activity.any():
                    act.start_activity(t, start_activity)
                    act.finalize_start(start_activity)

    def postpone_activities(self, population_selector, activity_queue):
        resource_not_available = self.check_activity_resource_availability(population_selector,
                                                                           activity_queue)

        ids = self.population.index[population_selector][resource_not_available]
        if resource_not_available.any():
            act_descriptor_specs = self.planned_activities[ids].pop()
            self.postponed_activities[ids].append(act_descriptor_specs)

        resource_available = np.ones(len(self.population), dtype=bool)
        resource_available[ids] = False
        return resource_available

    def check_activity_resource_availability(self, population_selector, activity_queue):
        """Returns which resources defined in the activity descriptor specifications of the activity_queue are
        occupied."""
        if type(population_selector) is not slice and not population_selector.any():
            return np.array([], dtype=bool)

        if len(self.location_ids) == 0:
            ids = self.population.index[population_selector]
            return np.zeros(len(ids), dtype=bool)

        blocks_location = activity_queue.blocks_location[population_selector].astype(bool)
        blocks_parent_location = activity_queue.block_parent_location[population_selector].astype(bool)

        # if the new activity is blocking, activity_queue, allow only the first instance
        occupied_resources = np.zeros(len(blocks_location), dtype=bool)

        # We need to verify that if someone is locking the parent, than the children will also be locked.
        blocked_location = self.location_blocked.copy()

        if blocks_parent_location.any():
            blocked_location = self.consider_children_block_state_before_blocking_the_parent(
                activity_queue, blocked_location, blocks_parent_location, population_selector)

        if blocks_location.any():
            next_activity_location = activity_queue.location_id[population_selector][blocks_location]
            location_mask = self.location_ids[:, 0] == next_activity_location.reshape(-1, 1)
            location_idx = np.where(location_mask)[1]
            locked_status = blocked_location[location_idx]
            first_come = enforce_unique_resource_utilization(location_mask, next_activity_location)
            occupied_resources[blocks_location] |= locked_status | first_come

        if (~blocks_location).any():
            next_activity_location = activity_queue.location_id[population_selector][~blocks_location]
            location_mask = self.location_ids[:, 0] == next_activity_location.reshape(-1, 1)
            occupied_resources[~blocks_location] |= (location_mask & blocked_location).any(axis=1)

        return occupied_resources

    def consider_children_block_state_before_blocking_the_parent(
            self, activity_queue, blocked_location, blocks_parent_location, population_selector):
        """ Ensure that if someone wants to block the parent then the parent's sub-locations are all free. Then block
        all sub-locations except for the one requested by  the first parent blocking agent."""
        # Select blocked parents
        next_activity_location = activity_queue.location_id[population_selector][blocks_parent_location]
        parents_idx = np.where(self.location_ids[:, 0] == next_activity_location.reshape(-1, 1))[1]
        next_activity_parents = self.location_ids[parents_idx, 1]
        parents = np.unique(next_activity_parents)
        # Update parents_idx_bool to point to the parent ids
        parents_idx_bool = (self.location_ids[:, 0] == parents.reshape(-1, 1)).any(axis=0)

        # Select all children of blocked parents
        # Create parent children association mask, TODO: Sparse matrix?
        children_idx = (self.location_ids[:, 1] == parents.reshape(-1, 1)).any(axis=0)
        location_pairs = self.location_ids[children_idx, :]
        lp_mask = (np.unique(location_pairs[:, 1]).reshape(-1, 1) == location_pairs[:, 1])

        # Mark parents as locked do to locked children
        blocked_location[parents_idx_bool] |= (lp_mask & blocked_location[children_idx]).any(axis=1)
        # Mark children as locked do to locked parents
        blocked_location[children_idx] |= (lp_mask.T & blocked_location[parents_idx_bool]).any(axis=1)

        # Select first agent blocking requesting to block the parent
        location_mask = self.location_ids[:, 0] == next_activity_parents.reshape(-1, 1)
        first_come = enforce_unique_resource_utilization(location_mask, next_activity_parents)
        unmask = ~(blocked_location[parents_idx].copy() | first_come)

        blocked_location[parents_idx_bool] = True
        # Mark children as locked do to locked parents
        blocked_location[children_idx] |= (lp_mask.T & blocked_location[parents_idx_bool]).any(axis=1)
        # Free first comers
        blocked_location[parents_idx[unmask]] = False

        return blocked_location

    def register_available_locations(self):
        if self.world is None:
            return

        world_regions = [[-1, -1, -1]]
        for r in self.world.list_all_regions():
            if r.parent is None:
                world_regions.append([r.id, -1, r])
            else:
                world_regions.append([r.id, r.parent.id, r])

        self.location_blocked = np.zeros(len(world_regions), dtype=bool)
        self.location_ids = np.array(world_regions)
        sort = np.argsort(self.location_ids[:, 0])
        self.location_ids = self.location_ids[sort]

    def block_locations(self, activity_descriptors):
        block_location = activity_descriptors[:, 6].astype(bool)
        block_parent_location = activity_descriptors[:, 7].astype(bool)

        # If the parent is blocked, ensure the location is also blocked
        block_location |= block_parent_location

        locations = activity_descriptors[block_location, 5]
        parent_locations = activity_descriptors[block_parent_location, 5]

        if len(locations) > 0:
            locations_selector = (self.location_ids[:, [0]] == locations).any(axis=1)  # type: ignore
            self.location_blocked[locations_selector] = True

        if len(parent_locations) > 0:
            locations_selector = (self.location_ids[:, [0]] == parent_locations).any(axis=1)  # type: ignore
            parent_ids = np.unique(self.location_ids[locations_selector, 1])

            # Block the parent
            locations_selector = (self.location_ids[:, [0]] == parent_ids).any(axis=1)  # type: ignore
            self.location_blocked[locations_selector] = True

            # Block the children
            locations_selector = (self.location_ids[:, [1]] == parent_ids).any(axis=1)  # type: ignore
            self.location_blocked[locations_selector] = True

    def unblock_locations(self, ids):
        if len(ids) == 0:
            return

        if not self.location_blocked.any():
            return

        if not hasattr(self.population, "location"):
            return

        regions = self.population.location[ids]
        mask = self.location_ids[:, [2]] == regions
        location_selector = mask.any(axis=1)
        if location_selector.any():
            self.location_blocked[location_selector] = False
            parents_ids = np.unique(self.location_ids[location_selector, 1])
            mask = (self.location_ids[:, [1]][location_selector] == parents_ids)
            parents_to_unblock = (~(mask & self.location_blocked[location_selector].reshape(-1, 1))).all(axis=0)
            if parents_to_unblock.any():
                parents_idx = (self.location_ids[:, [0]] == parents_ids[parents_to_unblock]).any(axis=1)
                self.location_blocked[parents_idx] = False

    def unblock_empty_locations(self):
        if not hasattr(self.population, "location"):
            return

        if self.location_blocked.any():
            regions = self.population.location.ravel()
            mask = self.location_ids[:, [2]] == regions
            location_selector = (~mask).all(axis=1)

            parent_ids = np.unique(self.location_ids[self.location_blocked, 1])
            blocked_parent_mask = (self.location_ids[:, [0]] == parent_ids) & self.location_blocked.reshape(-1, 1)
            blocked_parent_selector = blocked_parent_mask.any(axis=0)
            if blocked_parent_mask.any():
                blocked_parents_ids = parent_ids[blocked_parent_selector]
                children_empty_mask = (self.location_ids[:, [1]] == blocked_parents_ids) & ~location_selector.reshape(-1, 1)
                unblock_parent_ids = (~children_empty_mask).all(axis=0)
                keep_parent_blocked_selector = (self.location_ids[:, [0]] ==
                                                blocked_parents_ids[~unblock_parent_ids]).any(axis=1)
                keep_children_blocked_selector = (self.location_ids[:, [1]] ==
                                                  blocked_parents_ids[~unblock_parent_ids]).any(axis=1)
                location_selector &= ~(keep_children_blocked_selector | keep_parent_blocked_selector)

            self.location_blocked[location_selector] = False

    def remove_time_blocked_activities(self, have_new_activities):
        new_activity_types = self.planned_activities.queue[:, 0, 0]
        new_activity_types[new_activity_types == -1] = 0

        index_vector = np.array([np.arange(len(new_activity_types), dtype=int),
                                 np.full(len(new_activity_types), ActivityNone.blocked_for_ix),
                                 new_activity_types])
        index_vector = np.ravel_multi_index(index_vector, self.activities.activity_values.shape)
        blocked_for = self.activities.activity_values.ravel()[index_vector]
        discard = (blocked_for > 0) & have_new_activities
        if discard.any():
            self.planned_activities[discard].pop()

    def stop_activities_after_location_changed(self, t):
        if not hasattr(self.population, "location"):
            return

        changed_location = self.current_location_id != [loc.id for loc in self.population.location]
        changed_location &= self.current_location_id != -1
        if changed_location.any():
            self.stop_activities(changed_location, t)





