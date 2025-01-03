from functools import partial
from typing import Union, TYPE_CHECKING

import numpy as np

from i2mb import Model
from i2mb.activities import ActivityDescriptorProperties
from i2mb.activities.activity_descriptors import Rest
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs, ActivityDescriptor
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time, time

if TYPE_CHECKING:
    from i2mb.worlds import World, CompositeWorld
    from i2mb.activities.activity_manager import ActivityManager


def default_start_delay(size):
    return np.random.randint(0, global_time.make_time(minutes=15), size=size)


class LocationActivitiesController(Model):
    z_order = 0

    def __init__(self, population: AgentList, routine_schedules=None, start_delay=None):

        super().__init__()
        self.population = population

        # Operational Stuff
        # Index to keep track of descriptors per region
        self.location_activities_under_my_control = {}

        # Index the activities that are under our control, and their types
        self.controlled_activities = set()
        self.controlled_activities_types = set()

        # Pointer to available activities
        self.descriptor_index = np.empty(len(population), dtype=object)
        self.descriptor_index[:] = [[]]

        # Planner
        self.plan = ActivityDescriptorSpecs(size=len(population)).specifications
        if start_delay is None:
            start_delay = default_start_delay

        self.start_delay = start_delay

        # Track that the person has planned, started and finished an activity
        self.has_plan = np.zeros(len(self.population), dtype=bool)
        self.started = np.zeros(len(self.population), dtype=bool)
        self.finished = np.ones(len(self.population), dtype=bool)
        self.update_activity = np.zeros(len(self.population), dtype=bool)
        self.in_controlled_region = np.zeros(len(self.population), dtype=bool)

        # Routine Management
        self.routine_schedule = routine_schedules
        if routine_schedules is None:
            self.load_default_routines()

        self.routines_index = np.array(list(self.routine_schedule.keys()), ndmin=2)
        self.skip_activities = set()
        self.current_routine = None

    def registration_callback(self, activity_manager: 'ActivityManager', world: 'World'):
        self.register_enter_actions(activity_manager)
        self.__register_as_controller(activity_manager, world)
        self.register_on_activity_start()
        self.register_on_activity_stop()
        self.register_on_activity_unblock()

        regions = set(self.population.regions)
        for region in regions:
            self.notify_location_changes(region.population.index, region, [world] * len(region.population))

    def has_new_activity(self, inactive_ids):
        self.update_activity = self.has_plan.copy()
        self.update_activity &= (self.plan[:, ActivityDescriptorProperties.start] <= time())
        if ((self.plan[self.has_plan, ActivityDescriptorProperties.start] <= time())
             & ~self.update_activity[self.has_plan]).any():
            print(self.plan[self.has_plan, ActivityDescriptorProperties.start])

        self.update_activity &= ~self.started
        self.update_activity &= self.finished
        self.update_activity &= self.in_controlled_region
        return self.update_activity

    def get_new_activity(self, ids):
        if self.update_activity.any():
            act_specs = self.plan[ids, :]

            return act_specs[self.update_activity[ids]], self.update_activity[ids]

        return np.zeros_like(ids, dtype=object), np.zeros_like(ids, dtype=bool)

    def step(self, t):
        self.update_activity_plan(t)
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

    def update_routine_period(self, t):
        routine_period = self.get_routine_period(t)
        if not routine_period:
            self.current_routine = None
            return

        new_routine = self.routine_schedule[routine_period]["name"]
        if new_routine != self.current_routine:
            self.current_routine = new_routine
            self.skip_activities = self.routine_schedule[routine_period]["skip_activities"]

    def get_routine_period(self, t):
        mask = (self.routines_index[:, 0] < t) & (t <= self.routines_index[:, 1])
        # noinspection PyUnresolvedReferences
        return tuple(self.routines_index[mask.ravel(), :].flatten())

    def update_routine_activities(self, t):
        pass

    def stop_activity_callback(self, activity_id, t, stop_selector):
        # repopulate single agent descriptor queues
        return

    def stopped_activities(self, act_id, t, stop_selector):
        self.started[stop_selector] = False
        self.finished[stop_selector] = True

    def started_activities(self, act_id, t, start_selector):
        self.started[start_selector] = True
        self.has_plan[start_selector] = False
        act_specs = self.plan[start_selector]

        blocking_activities = act_specs[:, ActivityDescriptorProperties.block_for] > 0
        if blocking_activities.any():
            for available_activities in self.descriptor_index[start_selector][blocking_activities]:
                for ix in range(len(available_activities)):
                    descriptor = available_activities[ix]
                    if descriptor.activity_class.id == act_id:
                        del available_activities[ix]
                        break

    def unblocked_activities(self, act_id, t, start_selector):
        locations = self.population.location[start_selector]
        idx = self.population.index[start_selector]
        for location, agent in zip(locations, idx):
            location_index = location.index

            # use parent id
            if location.parent is not None and location.parent.available_activities is location.available_activities:
                location_index = location.parent.index

            if location_index in self.location_activities_under_my_control and \
                    act_id in [act.activity_class.id for act in self.location_activities_under_my_control[location_index]]:
                self.descriptor_index[[agent]] = [self.location_activities_under_my_control[location_index][::]]

    def notify_location_changes(self, ids, new_location, arriving_from):
        notify = np.ones_like(ids, dtype=bool)

        if new_location.parent is not None and new_location.parent.available_activities:
            # Activities are defined at the parent level, update only people coming from outside the parent
            arriving_parent = np.array([loc.parent for loc in arriving_from])
            notify &= (arriving_parent != new_location.parent).ravel() # noqa
            self.reset_region_available_activities(ids[notify])
            if new_location.parent.index in self.location_activities_under_my_control:
                location_descriptors = self.location_activities_under_my_control[new_location.parent.index]
                self.descriptor_index[ids[notify]] = [location_descriptors[::]]  # Each agent gets its own copy
                self.in_controlled_region[ids[notify]] = True

            return

        self.reset_region_available_activities(ids[notify])
        if new_location.index in self.location_activities_under_my_control:
            location_descriptors = self.location_activities_under_my_control[new_location.index]
            self.descriptor_index[ids[notify]] = [location_descriptors[::]]  # Each agent gets its own copy
            self.in_controlled_region[ids[notify]] = True

    def reset_region_available_activities(self, ids):
        self.plan[ids] = -1
        self.descriptor_index[ids] = [[]]
        self.has_plan[ids] = False
        self.in_controlled_region[ids] = False

    def register_exit_action(self):
        pass

    def register_on_activity_stop(self):
        for activity in self.controlled_activities:
            activity.register_stop_callbacks(self.stopped_activities)

    def register_on_activity_start(self):
        for activity in self.controlled_activities:
            activity.register_start_callbacks(self.started_activities)

    def register_on_activity_unblock(self):
        for activity in self.controlled_activities:
            activity.register_unblock_callbacks(self.unblocked_activities)

    def register_enter_actions(self, activity_manager):
        relocator = activity_manager.relocator
        relocator.register_on_region_enter_action(self.notify_location_changes)

    def __register_as_controller(self, activity_manager, world):
        processed_regions = set()
        for r in world.list_all_regions():
            if not hasattr(r, "available_activities") or len(r.available_activities) == 0:
                continue

            # we see the parent first, so we skipp the children. We assume that available_activities is synced between
            # parents and children.
            if r.parent in processed_regions:
                continue

            for act_descriptor in r.available_activities:
                if act_descriptor.activity_class.id in activity_manager.activity_controllers:
                    continue

                # activity_manager.add_location_activity_controller(*key, self)
                self.location_activities_under_my_control.setdefault(r.index, []).append(act_descriptor)
                activity = activity_manager.activity_list.activities[act_descriptor.activity_class.id]
                self.controlled_activities.add(activity)

            processed_regions.add(r)

    def update_activity_plan(self, t):
        un_planned = ~self.has_plan
        descriptors_available = np.array([len(ad) > 0 for ad in self.descriptor_index])
        un_planned &= descriptors_available
        if un_planned.any():
            new_plan = [np.random.choice(descriptors).create_specs()
                        for descriptors in self.descriptor_index[un_planned]]

            self.plan[un_planned, :] = ActivityDescriptorSpecs.merge_specs(new_plan).specifications
            delay = self.start_delay(un_planned.sum())
            self.plan[un_planned, ActivityDescriptorProperties.start] = delay + t
            self.has_plan[un_planned] = True



