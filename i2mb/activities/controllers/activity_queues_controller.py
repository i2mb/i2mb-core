import numpy as np

from i2mb import Model
from i2mb.activities import ActivityDescriptorProperties, ActivityProperties
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.base_activity import ActivityNone
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs, ActivityDescriptorQueue
from i2mb.engine.relocator import Relocator


def enforce_unique_resource_utilization(requested_location):
    """Given a list of location requests, this method will grant the first request per location.
    The response is a boolean array with the shape of requested_location."""

    def process_mask_column(column):
        response = np.zeros_like(requested_location)
        indexes = np.arange(len(response), dtype=int)
        response[indexes[column][0]] = True
        return response

    location_mask = requested_location.reshape(-1, 1) == np.unique(requested_location)
    return np.apply_along_axis(process_mask_column, 0, location_mask).sum(axis=1).astype(bool)


class ActivityQueueController(Model):
    file_headers = ["id", "activity", "start", "duration", "location"]

    def __init__(self, population, relocator: Relocator, activities: ActivityManager = None):

        super().__init__()
        if activities is None:
            activities = ActivityManager(population, relocator)

        self.file = None
        self.activity_manager = activities
        self.current_activity = self.activity_manager.current_activity
        self.population = population

        # Location occupancy management
        self.location_blocked = np.array([])
        self.region_index = np.array([[-1, -1, -1]])
        self.register_available_locations()
        self.current_location_id = np.zeros_like(self.activity_manager.current_activity, dtype=int)

        # Default activity can be changed during runtime
        #
        self.current_default_activity_descriptor = ActivityDescriptorSpecs(ActivityNone.id,
                                                                           size=len(self.population)).specifications
        self.current_default_activity = self.current_default_activity_descriptor[:,
                                                                                 ActivityDescriptorProperties.act_idx]
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
        self.register_activity_stop_logger()
        self.register_enter_actions()
        self.register_exit_action()

    def step(self, t):
        self.starting_activity[:] = False

        # Handle finished activities
        self.stage_default_activities(t)

        # handle new activities
        self.update_controllers(t)
        self.consume_interrupted_activities(t)
        self.consume_postponed_activities(t)
        self.consume_triggered_activities(t)
        self.consume_planned_activities(t)
        self.initiate_activities(t)

    def pause_activities(self, have_new_activities):
        # pause activities
        act_pause = have_new_activities.copy()
        act_pause &= self.activity_manager.get_current_activity_property(ActivityProperties.duration) > 0
        act_pause &= self.activity_manager.get_current_activity_property(ActivityProperties.blocked_for) == 0
        act_pause &= self.current_activity != self.current_default_activity
        if act_pause.any():
            paused_descriptors = convert_activities_to_descriptors(
                self.current_activity[act_pause],
                self.activity_manager.activity_values[act_pause, :, self.current_activity[act_pause]],
                self.current_location_id[act_pause],
                self.activity_manager.current_descriptors[act_pause],
            )

            self.interrupted_activities[act_pause].push(paused_descriptors)

    def stop_activities_on_request(self, stop_ids, t):
        self.unblock_locations(stop_ids)
        self.activity_manager.stop_activities(t, stop_ids)

        # Signal stopped activities to Controllers
        for controller in self.controllers:
            controller.stopped_activities(stop_ids)

    def generate_activity_ranking(self):
        for ix, act in enumerate(self.activity_manager.activity_manager):
            self.activity_ranking.setdefault(act.rank, list()).append((ix, act))

    def consume_interrupted_activities(self, t):
        has_interrupted_activities = self.interrupted_activities.num_items > 0

        # We opt not to test for in_progress, as the default activity should be interrupted to finish any
        # interrupted activities.
        current_activity_is_finished = (self.current_activity ==
                                        self.current_default_activity_descriptor[:,
                                        ActivityDescriptorProperties.act_idx])
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
        starting_activities = self.starting_activity
        starting_activities &= (self.current_activity != self.current_default_activity)
        starting_activities &= (self.current_activity != -1)
        have_new_activities = (self.planned_activities.num_items > 0) & ~starting_activities
        if have_new_activities.any():
            have_new_activities = self.remove_time_blocked_activities(have_new_activities)
            act_ready_to_start = (self.planned_activities.start <= t) & have_new_activities
            act_wait = self.activity_manager.get_current_activity_property(ActivityProperties.duration) > 0
            act_wait &= self.activity_manager.get_current_activity_property(ActivityProperties.blocked_for) > 0
            act_ready_to_start &= ~act_wait
            if act_ready_to_start.any():
                self.pause_activities(act_ready_to_start)
                not_postponed_act = self.postpone_activities(act_ready_to_start, self.planned_activities)
                # print(not_postponed_act)
                act_ready_to_start &= not_postponed_act
                if act_ready_to_start.any():
                    new_activities = self.planned_activities[act_ready_to_start].pop()
                    self.activity_manager.stop_activities(t, act_ready_to_start)
                    self.stage_activities(act_ready_to_start, new_activities, t)

        return

    def stage_activities(self, has_activities_to_stage, activity_descriptors_to_stage, t):
        # Start times
        activity_descriptors_to_stage[:, 1] = t

        # # Update Location for -1
        # no_location_change = activity_descriptors_to_stage[:, ActivityDescriptorProperties.location_ix] == -1
        # activity_descriptors_to_stage[no_location_change,
        #                               ActivityDescriptorProperties.location_ix] = self.current_location_id[has_activities_to_stage][
        #     no_location_change]

        ids = self.population.index[has_activities_to_stage]
        no_blocked_activities = self.activity_manager.stage_activity(activity_descriptors_to_stage, ids)
        idx = ids[no_blocked_activities]
        self.starting_activity[idx] = True

    def initiate_activities(self, t):
        if self.starting_activity.any():
            self.activity_manager.start_activities(self.starting_activity)

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

        if len(self.region_index) == 1:
            ids = self.population.index[population_selector]
            return np.zeros(len(ids), dtype=bool)

        blocks_location = activity_queue.blocks_location[population_selector].astype(bool)
        blocks_parent_location = activity_queue.block_parent_location[population_selector].astype(bool)

        # if the new activity is blocking, activity_queue, allow only the first instance
        occupied_resources = np.zeros(len(blocks_location), dtype=bool)

        # We need to verify that if someone is locking the parent, then the children will also be locked.
        blocked_location = self.location_blocked.copy()

        if blocks_parent_location.any():
            blocked_location = self.consider_children_block_state_before_blocking_the_parent(
                activity_queue, blocked_location, blocks_parent_location, population_selector)

        if blocks_location.any():
            next_activity_location = activity_queue.location_ix[population_selector][blocks_location]
            locked_status = blocked_location[next_activity_location]
            first_come = enforce_unique_resource_utilization(next_activity_location)
            occupied_resources[blocks_location] |= locked_status | ~first_come
            print(occupied_resources, locked_status, ~first_come)

        if (~blocks_location).any():
            next_activity_location = activity_queue.location_ix[population_selector][~blocks_location]
            occupied_resources[~blocks_location] |= blocked_location[next_activity_location]

        return occupied_resources

    def consider_children_block_state_before_blocking_the_parent(
            self, activity_queue, blocked_location, blocks_parent_location, population_selector):
        """ Ensure that if someone wants to block the parent then the parent's sub-locations are all free. Then block
        all sub-locations except for the one requested by  the first parent blocking agent."""
        # Select blocked parents
        next_activity_location = activity_queue.location_ix[population_selector][blocks_parent_location]
        next_activity_location = next_activity_location[next_activity_location != 0]
        parents_idx = self.region_index[next_activity_location, 1].astype(int)
        parents_idx = parents_idx[parents_idx != 0]
        parents = np.unique(parents_idx)

        # Update parents_idx_bool to point to the parent ids
        parents_idx_bool = np.zeros_like(blocked_location, dtype=bool)
        parents_idx_bool[parents_idx] = True

        # Select all children of blocked parents
        # Create parent children association mask, TODO: Sparse matrix?
        children_idx = (self.region_index[:, 1] == parents.reshape(-1, 1)).any(axis=0)
        location_pairs = self.region_index[children_idx, :]
        lp_mask = (np.unique(location_pairs[:, 1]).reshape(-1, 1) == location_pairs[:, 1])

        # Mark parents as locked do to locked children
        blocked_location[parents_idx_bool] |= (lp_mask & blocked_location[children_idx]).any(axis=1)
        # Mark children as locked do to locked parents
        blocked_location[children_idx] |= (lp_mask.T & blocked_location[parents_idx_bool]).any(axis=1)

        # Select first agent blocking requesting to block the parent
        first_come = enforce_unique_resource_utilization(parents_idx)
        unmask = ~(blocked_location[parents_idx].copy() | ~first_come)

        blocked_location[parents_idx_bool] = True
        # Mark children as locked do to locked parents
        blocked_location[children_idx] |= (lp_mask.T & blocked_location[parents_idx_bool]).any(axis=1)
        # Free first comers
        blocked_location[parents_idx[unmask]] = False

        return blocked_location

    def register_available_locations(self):
        if self.world is None:
            return

        self.region_index = self.world.region_index
        self.location_blocked = self.world.blocked_locations

    def remove_time_blocked_activities(self, have_new_activities):
        new_activity_types = self.planned_activities.queue[:, 0, 0]
        new_activity_types[new_activity_types == -1] = 0

        index_vector = np.array([np.arange(len(new_activity_types), dtype=int),
                                 np.full(len(new_activity_types), ActivityProperties.blocked_for),
                                 new_activity_types])
        index_vector = np.ravel_multi_index(index_vector, self.activity_manager.activity_values.shape)
        blocked_for = self.activity_manager.activity_values.ravel()[index_vector]
        discard = (blocked_for > 0) & have_new_activities
        if discard.any():
            self.planned_activities[discard].pop()
            have_new_activities[discard] = False

        return have_new_activities

    def log_finished_activities(self, activity_id, t, ids):
        if self.file is None:
            return

        if len(ids) == 0:
            return

        ids = self.population.index[ids]
        activity = self.activity_manager.activity_manager[activity_id]
        elapsed = activity.get_elapsed()[ids]
        start = activity.get_start()[ids]
        activity_type = [type(activity).__name__] * len(ids)
        location_ids = activity.get_location()[ids]  # type: np.ndarray
        location_ids_mask = (self.region_index[:, 0:1] == location_ids)  # type: np.ndarray
        location_types = np.where(location_ids_mask.T)[1]
        locations = [type(loc_).__name__ for loc_ in self.region_index[location_types, 2]]
        try:
            activity_log = np.vstack([ids, activity_type, start, elapsed, locations]).T
        except ValueError as e:
            print(self.region_index[:, 0:1] == location_ids)
            return

        self.file.write("\n".join([",".join(r) for r in activity_log]) + "\n")

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def register_activity_stop_logger(self):
        for activity in self.activity_manager.activity_manager:
            activity.register_stop_callbacks(self.log_finished_activities)

    def reset_queues_after_relocation(self, idx, location, global_population):
        idx = global_population.index[idx]
        self.planned_activities[idx].reset()
        self.postponed_activities[idx].reset()
        self.interrupted_activities[idx].reset()

    def update_current_location(self, n, idx, location, arrived_from):
        self.current_location_id[idx] = location.index

    def update_controllers(self, t):
        for controller in self.controllers:
            controller.fill_planned_activities_queue(self.planned_activities)

    # def update_activity_none_location(self, n, idx, location, arriving_from):
    #     ids = self.population.index[idx]
    #     self.activities.activities[0].get_location()[ids] = location.id

    def register_exit_action(self):
        for region in self.world.list_all_regions():
            if not isinstance(region, World):
                continue

            if region.parent == self.world:
                region.exit_actions.extend([self.reset_queues_after_relocation])

    def register_enter_actions(self):
        relocator = self.activity_manager.relocator
        if relocator is None:
            return

        relocator.register_on_region_enter_action([
            self.stage_default_activities_on_entry
        ])

    def stage_default_activities_on_entry(self, idx, region, arriving_from):
        no_planned_activity = ~self.starting_activity[idx]
        if no_planned_activity.any():
            default_descriptor = self.current_default_activity_descriptor[idx, :][no_planned_activity]
            self.activity_manager.stage_activity(default_descriptor, idx)

    def stage_default_activities(self, t):
        inactive = self.current_activity == -1
        if inactive.any():
            ids = self.population.index[inactive]
            default_descriptor = self.current_default_activity_descriptor[ids, :]
            self.activity_manager.stage_activity(default_descriptor, ids)
            self.starting_activity[inactive] = True
