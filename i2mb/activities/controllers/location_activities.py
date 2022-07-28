from typing import Union, TYPE_CHECKING

import numpy as np

from i2mb import Model
from i2mb.activities.activity_descriptors import Rest
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time


if TYPE_CHECKING:
    from i2mb.worlds import World, CompositeWorld
    from i2mb.activities.activity_manager import ActivityManager


class LocationActivitiesController(Model):
    def __init__(self, population: AgentList, activity_manager: 'ActivityManager', routine_schedules=None):

        super().__init__()
        self.activity_manager = activity_manager
        self.population = population

        self.location_descriptors = {}
        self.available_activities_in_location_queue = {}

        self.active_locations = np.zeros(len(self.population), dtype=int)

        # Determine that the person is finished with the activity
        self.finished = np.ones(len(self.population), dtype=bool)

        self.routine_schedule = routine_schedules
        if routine_schedules is None:
            self.load_default_routines()

        self.routines_index = np.array(list(self.routine_schedule.keys()), ndmin=2)
        self.skip_activities = set()
        self.current_routine = None

        self.location_parent_index = {}
        self.location_parents_ids = set()
        self.descriptor_index = {}
        self.descriptors_in_use = np.array([], dtype=bool)

    def post_init(self, base_file_name=None):
        self.register_enter_actions()

    def step(self, t):
        self.update_routine_period(t)

    def load_default_routines(self):
        # Some Time constants
        time_500, time_1100, time_1400, time_1700, time_2300 = [global_time.make_time(hour=h) for h in
                                                                [5, 11, 14, 17, 23]]
        self.routine_schedule = {
            (time_500, time_1100): {"name": "Morning routine",
                                    "skip_activities": set()},
            (time_1100, time_1400): {"name": "Lunch",
                                     "skip_activities": {Rest}},
            (time_1400, time_1700): {"name": "Afternoon routine",
                                     "skip_activities": set()},
            (time_1700, time_2300): {"name": "Evening routine",
                                     "skip_activities": set()}
        }

    def populate_activity_queues(self):
        for loc, activities in self.location_descriptors.items():
            self.populate_activity_queues_by_loc(activities, loc)

    def update_activity_queues(self):
        for loc in self.location_parents_ids:
            activities = self.select_activities_under_my_control(self.location_descriptors[loc])
            activities = set(activities)
            skip_activities = {act for act in activities if type(act) in self.skip_activities}

            ma_activities_set = activities - skip_activities

            # Reset queues
            self.available_activities_in_location_queue[loc][:] = None

            # Append new descriptors
            ma_activities = self.available_activities_in_location_queue[loc]
            ma_activities[:len(ma_activities_set)] = list(ma_activities_set)

    def populate_activity_queues_by_loc(self, descriptors, loc):
        loc_parent = self.location_parent_index.get(loc, -1)
        if loc_parent in self.available_activities_in_location_queue:
            # all children location use the  same queue
            ma_activities = self.available_activities_in_location_queue[loc_parent]
            self.available_activities_in_location_queue[loc] = ma_activities
            return

        ma_activities = self.select_activities_under_my_control(descriptors)
        self.available_activities_in_location_queue[loc] = np.array(ma_activities)

        if loc_parent > -1:
            # we ran into a child first. So also set the parent.
            self.available_activities_in_location_queue[loc_parent] = self.available_activities_in_location_queue[loc]

    def select_activities_under_my_control(self, descriptors):
        ma_activities = []
        for act in descriptors:
            if not self.activities_under_my_control[act.activity_id]:
                continue

            ma_activities.append(act)
        return ma_activities

    def update_routine_period(self, t):
        routine_period = self.get_routine_period(t)
        if not routine_period:
            self.current_routine = None
            return

        new_routine = self.routine_schedule[routine_period]["name"]
        if new_routine != self.current_routine:
            self.current_routine = new_routine
            self.skip_activities = self.routine_schedule[routine_period]["skip_activities"]
            self.update_activity_queues()

    def update_activities_under_my_control(self, controlled_activities):
        under_my_control = np.zeros_like(self.activity_manager.activity_types, dtype=bool)
        for ix, act_type in enumerate(self.activity_manager.activity_types):
            if act_type in controlled_activities:
                under_my_control[ix] = True

        return under_my_control

    def get_routine_period(self, t):
        mask = (self.routines_index[:, 0] < t) & (t <= self.routines_index[:, 1])
        # noinspection PyUnresolvedReferences
        return tuple(self.routines_index[mask.ravel(), :].flatten())

    def update_routine_activities(self, t):
        pass

    def assign_activities_per_location(self, queue, location, finished):
        request_activities = sum(finished)
        available_activities = self.available_activities_in_location_queue[location]
        if len(available_activities) == 0:
            return

        activity_descriptors = []
        m_queue_selector = available_activities != None  # noqa
        for act_descriptor in np.random.choice(available_activities[m_queue_selector], request_activities):
            activity_descriptors.append(act_descriptor.create_specs())

        activity_descriptors = ActivityDescriptorSpecs.merge_specs(activity_descriptors)
        self.finished[finished] = False
        self.descriptors_in_use[activity_descriptors.specifications[:, 8]] = True
        queue[finished].append(activity_descriptors)

    def stop_activity_callback(self, activity_id, t, stop_selector):
        # repopulate single agent descriptor queues
        return

    def stage_next_activity(self, region):
        """Calls activity_manager.stage_activity"""
        inactive = (self.activity_manager.current_activity == 0)
        finished = self.finished | inactive
        # finished = self.activity_manager.planned_activities.num_items == 0
        if finished.any():
            for location in np.unique(self.active_locations[finished]):
                finished_in_location = (self.active_locations == location) & finished
                if finished_in_location.any():
                    self.assign_activities_per_location(queue, location, finished_in_location)

    def stopped_activities(self, ids):
        self.finished[ids] = True

    def notify_location_changes(self, n, ids, new_location, arriving_from):
        if new_location.parent is not None and arriving_from.parent is not None:
            if arriving_from.parent.id == new_location.parent.id:
                return

        self.active_locations[ids] = new_location.id
        default_action = -1

        self.descriptor_index[ids] = new_location.ava


    def register_exit_action(self):
        pass

    def register_enter_actions(self):
        relocator = self.activity_manager.relocator
        relocator.register_on_region_enter_action(self.notify_location_changes)
