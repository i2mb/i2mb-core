from itertools import product
from typing import Union

import numpy as np

from i2mb import Model
from i2mb.activities.atomic_activities import Rest
from i2mb.activities.base_activity import ActivityList
from i2mb.activities.routines import Routine
from i2mb.utils import global_time


class LocationActivities(Model):
    def __init__(self, population, world, activities: Union[ActivityList, None] = None):
        if activities is None:
            activities = ActivityList(population)

        self.activities = activities
        self.world = world
        self.population = population

        self.current_activity = activities.current_activity

        controlled_activities = self.register_available_location_activities(world)
        self.activities_under_my_control = self.update_activities_under_my_control(controlled_activities)

        # Lets do one location specific, and then we generalize
        activity_indices = np.arange(len(self.activities_under_my_control))[self.activities_under_my_control]

        self.routine_schedule = {
            (5, 11): {"name": "Morning routine",
                      "skip_activities": None},
            (11, 14): {"name": "Lunch",
                       "skip_activities": None},
            (14, 17): {"name": "Afternoon routine",
                       "skip_activities": None},
            (17, 23): {"name": "Evening routine",
                       "skip_activities": None}
        }
        padding_activity = self.activities.activity_types.index(Rest)
        self.routines = Routine(len(self.population), activity_indices, activity_list=activities,
                                padding_index=padding_activity)
        self.routines_index = np.array(list(self.routine_schedule.keys()), ndmin=2)
        self.current_routine = None

    def register_available_location_activities(self, world):
        activities_under_my_control = []
        for r in world.list_all_regions():
            if not hasattr(r, "local_activities"):
                continue

            for act in r.local_activities:
                act_type = act.activity_class
                if act_type in self.activities.activity_types:
                    continue

                activities_under_my_control.append(act_type)
                self.activities.register(act_type(self.population))

        return activities_under_my_control

    def step(self, t):
        self.update_routine_period(t)
        self.schedule_next_activity_in_routine(t)
        self.reset_schedule_for_not_at_home(t)

    def update_routine_period(self, t):
        routine_period = self.get_routine_period(t)
        if not routine_period:
            self.current_routine = None
            return

        new_routine = self.routine_schedule[routine_period]["name"]
        if new_routine != self.current_routine:
            self.current_routine = new_routine
            skip_activities = self.routine_schedule[routine_period]["skip_activities"]
            self.routines.reset_routine_queue(skip_activities=skip_activities)
            print(self.current_routine)

    def schedule_next_activity_in_routine(self, t):
        inactive = (self.activities.current_activity == 0)
        inactive |= ~(self.activities.get_current_activity_property(self.activities.duration_ix) > 0)
        inactive |= ~self.activities.get_current_activity_property(self.activities.in_progress_ix).astype(bool)
        inactive &= self.population.at_home
        if inactive.any():
            next_activities = self.routines.get_next_activity(inactive)
            inactive = self.population.index[inactive]
            locations = self.population.location[inactive]

            unique_activities = np.unique(next_activities)
            unique_locations = set(locations)
            for act, loc in product(unique_activities, unique_locations):
                loc_mask = locations == loc
                act_mask = next_activities == act
                assign_to_activity = loc_mask & act_mask
                act_type = self.activities.activity_types[act]
                # TODO: Fix use a dictionary index instead
                available_act = list(filter(lambda x: x.activity_class == act_type and x.available,
                                            loc.available_activities))
                assignees = min(np.count_nonzero(assign_to_activity), len(available_act))
                if assignees > 0:
                    ids = inactive[assign_to_activity]
                    # TODO: use distribution on the activity descriptors
                    act_duration = np.random.randint(1, 5, assignees)
                    self.activities.activities[act].get_duration()[ids[:assignees]] = act_duration
                    self.activities.activities[act].get_start()[ids[:assignees]] = t
                    self.activities.activities[act].location[ids[:assignees]] = [a_d.location for a_d in
                                                                                 available_act[:assignees]]
                    activity_positions = np.vstack([a_d.device.get_activity_position(pos_id=a_d.pos_id) for a_d in
                                                    available_act[
                                                    :assignees]])
                    self.activities.activities[act].device[ids[:assignees], :] = activity_positions
                    self.activities.activities[act].descriptor[ids[:assignees]] = available_act[:assignees]
                    for activity_descriptor in available_act[:assignees]:
                        activity_descriptor.available = False

    def reset_schedule_for_not_at_home(self, t):
        left_home = ~self.population.at_home
        if left_home.any():
            start_ix = self.activities.start_ix
            duration_ix = self.activities.duration_ix
            ids = self.population.index[left_home]
            index_vector = np.array(tuple(product(ids, [start_ix, duration_ix], self.activities_under_my_control)))
            index_vector = np.ravel_multi_index(index_vector.T, self.activities.activity_values.shape)
            self.activities.activity_values.ravel()[index_vector] = 0

    def update_activities_under_my_control(self, controlled_activities):
        under_my_control = np.zeros_like(self.activities.activity_types, dtype=bool)
        for ix, act_type in enumerate(self.activities.activity_types):
            if act_type in controlled_activities:
                under_my_control[ix] = True

        return under_my_control

    def get_routine_period(self, t):
        h = global_time.hour(t)

        mask = (self.routines_index[:, 0] < h) & (h <= self.routines_index[:, 1])
        # noinspection PyUnresolvedReferences
        return tuple(self.routines_index[mask.ravel(), :].flatten())

