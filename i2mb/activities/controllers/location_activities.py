from itertools import product
from typing import Union

import numpy as np

from i2mb import Model
from i2mb.activities.activity_descriptors import Rest
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time


class LocationActivitiesController(Model):
    def __init__(self, population: AgentList, world, activity_manager: Union[ActivityManager, None] = None,
                 routine_schedules=None):

        if activity_manager is None:
            activity_manager = ActivityManager(population, world)

        self.activity_manager = activity_manager
        self.world = world
        self.population = population

        self.current_activity = activity_manager.current_activity

        self.reset_location_activities = np.ones((len(self.population), 1), dtype=bool)
        self.population.add_property("reset_location_activities", self.reset_location_activities)
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
        controlled_activities = self.register_available_location_activities(world)
        self.activities_under_my_control = self.update_activities_under_my_control(controlled_activities)
        self.populate_activity_queues()

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

    def register_available_location_activities(self, world):
        activities_under_my_control = []
        for r in world.list_all_regions():
            if not hasattr(r, "local_activities"):
                continue

            if r.parent is not None and r.available_activities is r.parent.available_activities:
                self.location_parent_index[r.id] = r.parent.id
                self.location_parents_ids.add(r.parent.id)

            self.location_descriptors[r.id] = r.available_activities
            for act in r.local_activities:
                self.descriptor_index.setdefault(act.descriptor_id, act)
                act_type = act.activity_class
                if act_type not in self.activity_manager.activities.activity_types:
                    activities_under_my_control.append(act_type)
                    activity = act_type(self.population)
                    self.activity_manager.activities.register(activity)
                    activity.register_stop_callbacks(self.stop_activity_callback)

                act.activity_id = self.activity_manager.activities.activity_types.index(act_type)

        min_id = min(self.descriptor_index)
        max_id = max(self.descriptor_index)
        self.descriptors_in_use = np.zeros(max_id+1, dtype=bool)

        return activities_under_my_control

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

        return

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

    def step(self, t):
        self.update_local_activities(t)
        self.update_routine_period(t)
        self.schedule_next_activity_in_routine(t)
        # self.reset_schedule_for_not_at_home(t)

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

    def schedule_next_activity_in_routine(self, t):
        inactive = (self.activity_manager.current_activity == 0)
        finished = self.finished | inactive
        finished = self.activity_manager.planned_activities.num_items == 0
        if finished.any():
            for location in np.unique(self.active_locations[finished]):
                finished_in_location = (self.active_locations == location) & finished
                if finished_in_location.any():
                    self.assign_activities_per_location(location, finished_in_location)

    def reset_schedule_for_not_at_home(self, t):
        left_home = ~self.population.at_home
        if left_home.any():
            start_ix = self.activity_manager.activities.start_ix
            duration_ix = self.activity_manager.activities.duration_ix
            ids = self.population.index[left_home]
            index_vector = np.array(tuple(product(ids, [start_ix, duration_ix], self.activities_under_my_control)))
            index_vector = np.ravel_multi_index(index_vector.T, self.activity_manager.activities.activity_values.shape)
            self.activity_manager.activities.activity_values.ravel()[index_vector] = 0

    def update_activities_under_my_control(self, controlled_activities):
        under_my_control = np.zeros_like(self.activity_manager.activities.activity_types, dtype=bool)
        for ix, act_type in enumerate(self.activity_manager.activities.activity_types):
            if act_type in controlled_activities:
                under_my_control[ix] = True

        return under_my_control

    def get_routine_period(self, t):
        mask = (self.routines_index[:, 0] < t) & (t <= self.routines_index[:, 1])
        # noinspection PyUnresolvedReferences
        return tuple(self.routines_index[mask.ravel(), :].flatten())

    def update_local_activities(self, t):
        if self.reset_location_activities.any():
            self.reset_queues_activities_under_my_control()
            new_locations = self.population.location[self.reset_location_activities.ravel()]
            new_location_ids = [r.id for r in new_locations]
            self.active_locations[self.reset_location_activities.ravel()] = new_location_ids
            self.finished[self.reset_location_activities.ravel()] = True
            self.reset_location_activities[:] = False

    def update_routine_activities(self, t):
        pass

    def assign_activities_per_location(self, location, finished):
        request_activities = sum(finished)
        available_activities = self.available_activities_in_location_queue[location]
        if len(available_activities) == 0:
            return

        activity_descriptors = []
        m_queue_selector = available_activities != None
        for act_descriptor in np.random.choice(available_activities[m_queue_selector], request_activities):
            activity_descriptors.append(act_descriptor.create_specs())

        activity_descriptors = ActivityDescriptorSpecs.merge_specs(activity_descriptors)
        self.finished[finished] = False
        self.descriptors_in_use[activity_descriptors.specifications[:, 8]] = True
        self.activity_manager.planned_activities[finished].append(activity_descriptors)

    def stop_activity_callback(self, activity_id, t, stop_selector, descriptors):
        # repopulate single agent descriptor queues
        return

    def reset_queues_activities_under_my_control(self):
        self.activity_manager.planned_activities[self.reset_location_activities.ravel()].reset()
        self.activity_manager.postponed_activities[self.reset_location_activities.ravel()].reset()
        self.activity_manager.interrupted_activities[self.reset_location_activities.ravel()].reset()
