#  dct_mct_analysis
#  Copyright (C) 2021  FAU - RKI
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
from time import sleep
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from i2mb.activities.controllers.default_activity_controller import DefaultActivityController
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.controllers.location_activities import LocationActivitiesController
from i2mb.activities.controllers.sleep import SleepBehaviourController
from i2mb.utils import global_time
from i2mb.worlds import Apartment, Bathroom
from tests.i2mb_test_case import I2MBTestCase
from tests.world_tester import WorldBuilder

global_time.ticks_hour = 60 // 5


class TestLocationActivityController(I2MBTestCase):
    def setup_engine(self, callbacks=None, no_gui=True, sleep=False, use_office=False):
        self.w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=5), sim_duration=global_time.make_time(day=3),
                              update_callback=callbacks, no_gui=no_gui, use_office=use_office)
        print(f"Running for {global_time.make_time(day=3)} ticks")

        self.population = self.w.population
        self.activity_manager = ActivityManager(self.w.population)

        # sleep_duration = partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(hour=1))
        # sleep_midpoint = partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(hour=1))
        def sleep_duration(shape):
            return np.ones(shape, dtype=int) * global_time.make_time(hour=8)

        def sleep_midpoint(shape):
            return np.ones(shape, dtype=int) * global_time.make_time(hour=1)

        self.activity_manager = ActivityManager(self.w.population, self.w.relocator)
        sleep_model = SleepBehaviourController(self.w.population, self.activity_manager, self.w.universe,
                                               sleep_duration,
                                               sleep_midpoint)

        self.location_manager = LocationActivitiesController(self.population,
                                                             activity_manager=self.activity_manager)

        self.sleep_model = sleep_model
        if sleep:
            self.w.engine.models.extend([sleep_model, self.location_manager, self.activity_manager])
        else:
            self.w.engine.models.extend([self.location_manager, self.activity_manager])

        self.w.engine.post_init_modules()
        self.engine_iterator = self.w.engine.step()

    def walk_engine(self, num_steps):
        t = self.w.engine.time
        for frame, rs in enumerate(self.engine_iterator, start=t):
            stop = self.w.process_stop_criteria(frame)
            if stop:
                return

            num_steps -= 1
            if num_steps <= 0:
                return

            self.w.update_callback(self.w, frame)

    def test_setup(self):
        self.setup_engine()
        expected_activities = [False, False] + [True for _ in self.activity_manager.activity_list.activities[2:]]
        current_activities = list(self.location_manager.activities_under_my_control)
        self.assertListEqual(expected_activities, current_activities)

        # Check that there are no activities registered for the square location
        loc_id = self.w.universe.regions[2].id
        self.assertListEqual([], self.location_manager.location_descriptors[loc_id])

        # Check activities registered for Apartment 0 and 1
        for i in range(2):
            loc_id = self.w.universe.regions[i].id
            available_activities = self.w.universe.regions[i].available_activities
            self.assertListEqual(available_activities, self.location_manager.location_descriptors[loc_id])

        # test that descriptor activity Id, match activity id
        for k, v in self.location_manager.location_descriptors.items():
            for act_descriptor in v:
                act_id = act_descriptor.activity_id
                expected_act_type = act_descriptor.activity_class
                act_type = self.activity_manager.activity_manager.activity_types[act_id]
                location = (self.activity_manager.region_index[:, 0] == k)
                location = self.activity_manager.region_index[location, 2]
                self.assertEqual(expected_act_type, act_type, msg=f"Problem found at {k}, {location}, {act_id}")

    def test_update_local_activities(self):
        self.setup_engine()
        expected_activities = [False, False] + [True for _ in self.activity_manager.activity_manager.activity_manager[2:]]
        current_activities = list(self.location_manager.activities_under_my_control)
        self.assertListEqual(expected_activities, current_activities)

        # Move agents to the square space
        self.w.universe.move_agents(slice(None), self.w.universe.regions[2])

        # Check that every one was reset
        self.assertTrue(self.activity_manager.reset_location_activities.all())

        # update_local_activities
        locations = self.population.location[self.activity_manager.reset_location_activities.ravel()]
        self.location_manager.notify_location_changes(self.activity_manager.reset_location_activities.ravel(),
                                                      locations)

        # active location
        self.assertTrue((self.location_manager.active_locations == self.w.universe.regions[2].id).all())

        for home in self.w.universe.regions[:2]:
            self.w.universe.move_agents(self.population.home == home, home)

        self.assertTrue(self.activity_manager.reset_location_activities.all())
        locations = self.population.location[self.activity_manager.reset_location_activities.ravel()]
        self.location_manager.notify_location_changes(self.activity_manager.reset_location_activities.ravel(),
                                                      locations)

        expected_location = [self.w.universe.regions[0].get_entrance_sub_region().id] * 5 + \
                            [self.w.universe.regions[1].get_entrance_sub_region().id] * 5
        actual_location = list(self.location_manager.active_locations)
        self.assertListEqual(expected_location, actual_location)

    def test_update_routine_period(self):
        self.setup_engine()
        time_500, time_1100, time_1400, time_1700, time_2300 = [global_time.make_time(hour=h)
                                                                for h in [5, 11, 14, 17, 23]]
        for t in range(global_time.make_time(day=4)):
            self.location_manager.update_routine_period(t)
            if t < time_500:
                self.assertEqual(None, self.location_manager.current_routine, msg=f"{t}, {t}")

            if time_500 < t < time_1100:
                self.assertEqual("Morning routine", self.location_manager.current_routine,
                                 msg=f"{time_500}, {t}, {time_1100}")

            if time_1100 < t < time_1400:
                self.assertEqual("Lunch", self.location_manager.current_routine, msg=f"{t}, {t}")
                routine = (time_1100, time_1400)
                for loc, queue in self.location_manager.available_activities_in_location_queue.items():
                    skip_set = self.location_manager.routine_schedule[routine]["skip_activities"]
                    test_set = {type(a) for a in queue}
                    self.assertListEqual([], list(test_set.intersection(skip_set)))

            if time_1400 < t < time_1700:
                self.assertEqual("Afternoon routine", self.location_manager.current_routine, msg=f"{t}, {t}")

            if time_1700 < t < time_2300:
                self.assertEqual("Evening routine", self.location_manager.current_routine, msg=f"{t}, {t}")

    def test_schedule_activities(self):
        self.setup_engine()
        self.location_manager.update_local_activities(1)
        self.location_manager.schedule_next_activity_in_routine()

        self.assertTrue((self.activity_manager.planned_activities.queue[:, 0, 0] > 1).all())

    def test_activity_assignment_per_location(self):
        self.setup_engine()
        finished = [True, False] * 5

        for i in range(3):
            for location in set(self.population.location):
                finished_in_location = (self.population.location == location) & finished
                if finished_in_location.any():
                    self.location_manager.assign_activities_per_location(self.activity_manager.planned_activities,
                                                                         location.id, finished_in_location)

        self.assertListEqual([3 * i for i in finished], list(self.activity_manager.planned_activities.num_items),
                             msg=f"Num_Items={self.activity_manager.planned_activities.num_items}")

        # return the stop_activity_callback, to check that all activities come back
        for i in range(3):
            descriptor_specs = self.activity_manager.planned_activities[finished].pop()
            self.location_manager.stop_activity_callback(None, 0, None, descriptor_specs[:, 8])

    def test_location_manager_stop(self):
        def callback(world_builder, frame):
            time_500, time_1100, time_1400, time_1700, time_2300 = [global_time.make_time(hour=h)
                                                                    for h in [5, 11, 14, 17, 23]]

            if time_1100 < frame < time_1400:
                print(frame)
                self.assertEqual("Lunch", self.location_manager.current_routine, msg=f"{frame}, {frame}")

        self.setup_engine(callback)
        bathrooms = [loc for loc in self.w.universe.list_all_regions() if type(loc) is Bathroom]
        for br in bathrooms:
            br.post_init()
        self.walk_engine(500)

    def test_location_manager_with_motion_to_empty_space(self):
        def callback(world_builder, frame):
            time_500, time_1100, time_1400, time_1700, time_2300 = [global_time.make_time(hour=h)
                                                                    for h in [5, 11, 14, 17, 23]]

            if time_1100 + 5 == frame:
                self.w.universe.move_agents(slice(0, 5), self.w.universe.regions[2])

            if time_1400 + 5 == frame:
                self.w.universe.move_agents(slice(0, 5), self.w.universe.regions[0])

            if time_1100 + 5 < frame < time_1400 + 5:
                self.assertListEqual([self.w.universe.regions[2]] * 5, list(self.population.location[slice(0, 5)]),
                                     msg=f"Error at {frame}")

        self.setup_engine(callback)
        bathrooms = [loc for loc in self.w.universe.list_all_regions() if type(loc) is Bathroom]
        for br in bathrooms:
            br.post_init()
        self.walk_engine(500)

    def check_if_queues_are_ever_empty(self):
        for loc, desc_list in self.location_manager.location_single_agent_available_activity_queue.items():
            if len(desc_list) == 0:
                continue

            queue_empty = (desc_list == None).all()
            if queue_empty:
                print("Single Queue Empty", loc)

    def test_location_manager_stop_gui(self):

        def callback(world_builder, frame):
            for br in bathrooms:
                selector = self.activity_manager.current_location_id == br.id
                activity_ids = self.activity_manager.current_activity[selector]
                activity_types = np.array(self.activity_manager.activity_manager.activity_types)[activity_ids]
                print([str(at).split(".")[-1][:-2] for at in activity_types])


        self.setup_engine(no_gui=False, sleep=False, callbacks=callback)

        bathrooms = [loc for loc in self.w.universe.list_all_regions() if type(loc) is Bathroom]
        for br in bathrooms:
            br.post_init()

        plt.show()

    def test_location_manager_with_sleep_stop_gui(self):
        finished = False
        def callback(world_builder, frame):
            time_500, time_1100, time_1400, time_1700, time_2300 = [global_time.make_time(hour=h)
                                                                    for h in [5, 11, 14, 17, 23]]
            for br in bathrooms:
                selector = self.activity_manager.current_location_id == br.id
                if selector.any():
                    activity_ids = self.activity_manager.current_activity[selector]
                    print(frame)
                    activity_types = np.array(self.activity_manager.activity_manager.activity_types)[activity_ids]
                    print([str(at).split(".")[-1][:-2] for at in activity_types], br.population.index)

            hour = global_time.to_current(time_1100, frame)
            if hour + 5 == frame:
                self.w.universe.move_agents(slice(0, 5), self.w.universe.regions[2])

            hour = global_time.to_current(time_1400, frame)
            if hour + 5 == frame:
                self.w.universe.move_agents(slice(0, 5), self.w.universe.regions[0])

            select = self.activity_manager.interrupted_activities.num_items > 0
            activities = ""
            if select.any():
                activities = self.activity_manager.interrupted_activities.queue[select, 0, 0]
                activities = np.array(self.activity_manager.activity_manager.activity_types)[activities]
                activities = ", ".join([str(a).split(".")[-1][:-2] for a in activities])

            self.assertFalse("Sleep" in activities)

        self.setup_engine(no_gui=False, sleep=True, callbacks=callback)
        bathrooms = [loc for loc in self.w.universe.list_all_regions() if type(loc) is Bathroom]
        for br in bathrooms:
            br.post_init()

        plt.show()

    def test_location_manager_with_sleep_office_gui(self):
        def callback(world_builder, frame):
            time_500, time_800, time_1100, time_1400, time_1700, time_2300 = [global_time.make_time(hour=h)
                                                                    for h in [5, 8, 11, 14, 17, 23]]
            for br in bathrooms:
                selector = self.activity_manager.current_location_id == br.id
                if selector.any():
                    activity_ids = self.activity_manager.current_activity[selector]
                    print(frame)
                    activity_types = np.array(self.activity_manager.activity_manager.activity_types)[activity_ids]
                    print([str(at).split(".")[-1][:-2] for at in activity_types], br.population.index)

            hour = global_time.to_current(time_800, frame)
            if hour + 5 == frame:
                self.w.universe.move_agents(slice(0, 3), self.w.universe.regions[2])
                self.w.universe.move_agents(slice(5, 8), self.w.universe.regions[2])

            hour = global_time.to_current(time_1700, frame)
            if hour + 5 == frame:
                self.w.universe.move_agents(slice(0, 3), self.w.universe.regions[0])
                self.w.universe.move_agents(slice(5, 8), self.w.universe.regions[1])

            select = self.activity_manager.interrupted_activities.num_items > 0
            activities = ""
            if select.any():
                activities = self.activity_manager.interrupted_activities.queue[select, 0, 0]
                activities = np.array(self.activity_manager.activity_manager.activity_types)[activities]
                activities = ", ".join([str(a).split(".")[-1][:-2] for a in activities])

            self.assertFalse("Sleep" in activities, msg=activities)

        self.setup_engine(no_gui=False, sleep=True, callbacks=callback, use_office=True)
        bathrooms = [loc for loc in self.w.universe.list_all_regions() if type(loc) is Bathroom]
        for br in bathrooms:
            br.post_init()

        plt.show()

    def test_location_manager_with_sleep_office_no_gui(self):
        def callback(world_builder, frame):
            time_500, time_800, time_1100, time_1400, time_1700, time_2300 = [global_time.make_time(hour=h)
                                                                    for h in [5, 8, 11, 14, 17, 23]]
            for br in bathrooms:
                selector = self.activity_manager.current_location_id == br.id
                if selector.any():
                    activity_ids = self.activity_manager.current_activity[selector]
                    print(frame)
                    activity_types = np.array(self.activity_manager.activity_manager.activity_types)[activity_ids]
                    print([str(at).split(".")[-1][:-2] for at in activity_types], br.population.index)

            hour = global_time.to_current(time_800, frame)
            if hour + 5 == frame:
                self.w.universe.move_agents(slice(0, 3), self.w.universe.regions[2])
                self.w.universe.move_agents(slice(5, 8), self.w.universe.regions[2])

            hour = global_time.to_current(time_1700, frame)
            if hour + 5 == frame:
                self.w.universe.move_agents(slice(0, 3), self.w.universe.regions[0])
                self.w.universe.move_agents(slice(5, 8), self.w.universe.regions[1])

            select = self.activity_manager.interrupted_activities.num_items > 0
            activities = ""
            if select.any():
                activities = self.activity_manager.interrupted_activities.queue[select, 0, 0]
                activities = np.array(self.activity_manager.activity_manager.activity_types)[activities]
                activities = ", ".join([str(a).split(".")[-1][:-2] for a in activities])

            self.assertFalse("Sleep" in activities)

        self.setup_engine(no_gui=True, sleep=True, callbacks=callback, use_office=True)
        bathrooms = [loc for loc in self.w.universe.list_all_regions() if type(loc) is Bathroom]
        for br in bathrooms:
            br.post_init()

        self.walk_engine(500)
