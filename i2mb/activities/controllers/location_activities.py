from typing import Union, TYPE_CHECKING

import numpy as np

from i2mb import Model
from i2mb.activities.activity_descriptors import Rest
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs, ActivityDescriptor
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time


if TYPE_CHECKING:
    from i2mb.worlds import World, CompositeWorld
    from i2mb.activities.activity_manager import ActivityManager


class LocationActivitiesController(Model):
    def __init__(self, population: AgentList, routine_schedules=None):

        super().__init__()
        self.population = population

        # Operational Stuff
        # Index to keep track of descriptors per region
        self.location_activities_under_my_control = {}

        # Index the activities that are under our control, and their types
        self.controlled_activities = set()
        self.controlled_activities_types = set()

        # Pointer to available activities
        self.descriptor_index = np.full(len(population), -1, dtype=object)
        self.descriptor_index[:] = [[]]

        # Track that the person has planned, started and finished an activity
        self.has_plan = np.zeros(len(self.population), dtype=bool)
        self.started = np.zeros(len(self.population), dtype=bool)
        self.finished = np.ones(len(self.population), dtype=bool)

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

    def step_on_handler(self, region):
        idx = region.population.index
        update_activity = ~self.has_plan[idx]
        update_activity |= ~self.started[idx]
        update_activity &= self.finished[idx]
        if update_activity.any():
            descriptors = self.location_activities_under_my_control[region.index]
            act_descriptors = np.random.choice(descriptors, update_activity.sum())
            act_specs = ActivityDescriptorSpecs.merge_specs([desc.create_specs() for desc in act_descriptors])
            self.has_plan[idx[update_activity]] = True
            self.started[idx[update_activity]] = False
            return act_specs.specifications, idx[update_activity]

        return [], []

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
        self.finished[stop_selector] = True

    def started_activities(self, act_id, t, stop_selector):
        self.started[stop_selector] = True
        self.has_plan[stop_selector] = False

    def notify_location_changes(self, ids, new_location, arriving_from):
        notify = np.ones_like(ids, dtype=bool)
        if new_location.parent is not None:
            # Activities are defined at the parent level, update only people coming from outside the parent
            arriving_parent = np.array([loc.parent for loc in arriving_from])
            notify &= (arriving_parent != new_location.parent).ravel() # noqa

            if new_location.parent.index in self.location_activities_under_my_control:
                location_descriptors = self.location_activities_under_my_control[new_location.parent.index]
                self.descriptor_index[ids[notify]] = [location_descriptors]
                return

        if new_location.index in self.location_activities_under_my_control:
            location_descriptors = self.location_activities_under_my_control[new_location.parent.index]
            self.descriptor_index[ids[notify]] = [location_descriptors]

    def register_exit_action(self):
        pass

    def register_on_activity_stop(self):
        for activity in self.controlled_activities:
            activity.register_stop_callbacks(self.stopped_activities)

    def register_on_activity_start(self):
        for activity in self.controlled_activities:
            activity.register_start_callbacks(self.started_activities)

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
                key = (r.index, act_descriptor.activity_class.id)
                if activity_manager.has_location_controller(*key):
                    continue

                activity_manager.add_location_activity_controller(*key, self)
                self.location_activities_under_my_control.setdefault(r.index, []).append(act_descriptor)
                activity = activity_manager.activity_list.activities[act_descriptor.activity_class.id]
                self.controlled_activities.add(activity)

            processed_regions.add(r)


