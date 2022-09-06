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
import numpy as np
from matplotlib import pyplot as plt

from i2mb.activities import ActivityDescriptorProperties
from i2mb.activities.activity_descriptors import Toilet
from i2mb.activities.atomic_activities import Sleep, Rest
from i2mb.activities.controllers.default_activity_controller import DefaultActivityController
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.controllers.location_activities import LocationActivitiesController
from i2mb.activities.controllers.sleep_controller import SleepBehaviourController
from i2mb.utils import global_time
from i2mb.worlds import Apartment, Bathroom
from tests.i2mb_test_case import I2MBTestCase
from tests.world_tester import WorldBuilder

global_time.ticks_hour = 60 // 5


class TestLocationActivityController(I2MBTestCase):
    def setup_engine(self, callbacks=None, no_gui=True, default=False, sleep=False, use_office=False):
        self.w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=5), sim_duration=global_time.make_time(day=3),
                              update_callback=callbacks, no_gui=no_gui, use_office=use_office)

        self.population = self.w.population
        self.activity_manager = ActivityManager(self.w.population)

        # sleep_duration = partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(hour=1))
        # sleep_midpoint = partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(hour=1))
        def sleep_duration(shape):
            return np.ones(shape, dtype=int) * global_time.make_time(hour=8)

        def sleep_midpoint(shape):
            return np.ones(shape, dtype=int) * global_time.make_time(hour=1)

        self.activity_manager = ActivityManager(self.w.population, self.w.relocator)
        self.sleep_model = SleepBehaviourController(self.w.population, self.activity_manager, sleep_duration, sleep_midpoint)
        self.default_activity_controller = DefaultActivityController(self.population)
        self.location_manager = LocationActivitiesController(self.population)

        self.w.engine.models.extend([self.location_manager, self.activity_manager])
        if sleep:
            self.activity_manager.register_activity_controller(self.sleep_model)
            self.w.engine.models.insert(0, self.sleep_model)

        if default:
            self.activity_manager.register_activity_controller(self.default_activity_controller, z_order=4)

        self.activity_manager.register_activity_controller(self.location_manager)

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


class TestLocationActivityControllerGui(TestLocationActivityController):
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


class TestLocationActivityControllerNoGui(TestLocationActivityController):
    def test_setup(self):
        self.setup_engine(sleep=True, default=True)

        for loc, activities in self.location_manager.location_activities_under_my_control.items():
            if isinstance(loc, Apartment):
                self.assertGreater(len(activities), 0)

            for act in activities:
                self.assertNotEqual(act.activity_class.id, Sleep.id, msg="Sleep should not be in the controller list")
                self.assertNotEqual(act.activity_class.id, Rest.id, msg="Rest should not be in the controller list")

        self.assertGreater(len(self.location_manager.controlled_activities), 0,
                           msg=f"{self.location_manager.controlled_activities}")

        self.assertListEqual(self.activity_manager.controllers,
                             [self.sleep_model,
                              self.location_manager,
                              self.default_activity_controller])

    def test_update_local_activities(self):
        self.setup_engine(sleep=True, default=True, use_office=True)

        # Move agents to the office space
        self.w.relocator.move_agents(slice(None), self.w.universe.regions[2])

        # check, no action for the controller
        expected = np.full(len(self.population), -1, dtype=object)
        coffee_break = self.w.universe.regions[2].available_activities[1]
        expected[:] = [[coffee_break]]
        self.assertEqualAll(self.location_manager.descriptor_index, expected)

        # Move People Back Home
        for home in self.w.universe.regions[:2]:
            self.w.relocator.move_agents(self.population.home == home, home)

        # Check that descriptor_index is not empty
        expected = np.full(len(self.population), -1, dtype=object)
        expected[:] = [[]]
        self.assertNotEqualAll(self.location_manager.descriptor_index, expected)

    def test_assign_activities(self):
        self.setup_engine()
        self.location_manager.start_delay = lambda x: np.full(x, 0, dtype=int)

        self.location_manager.update_activity_plan(0)
        self.location_manager.has_new_activity(self.population.index)

        # Agents start at home, so we can manually call the update to see next activity.
        for apartment in self.w.universe.regions[:2]:
            descriptors, idx = self.location_manager.get_new_activity(apartment.inhabitants.index)
            for act_id in descriptors[:, ActivityDescriptorProperties.act_idx]:
                activity = self.activity_manager.activity_list.activities[act_id]
                self.assertTrue(activity in self.location_manager.controlled_activities,
                                msg=f"{act_id} {activity} {self.location_manager.controlled_activities}")

        self.assertTrueAll(self.location_manager.has_plan)

    def test_start_activities(self):
        self.setup_engine()
        self.location_manager.start_delay = lambda x: np.full(x, 0, dtype=int)

        # Move a bit
        self.walk_engine(1)

        # Test that every one has started a location activity.
        self.assertTrueAll(self.location_manager.started)
        for act_id in self.activity_manager.current_activity:
            activity = self.activity_manager.activity_list.activities[act_id]
            self.assertTrue(activity in self.location_manager.controlled_activities, msg=f"{act_id} {activity}")

        self.assertTrueAll(self.location_manager.started)

    def test_respect_sleep(self):
        self.setup_engine(sleep=True, default=True, use_office=True)

        self.walk_engine(1)
        self.assertTrueAll(self.sleep_model.has_plan)
        self.assertLessAll(self.sleep_model.sleep_profiles.specifications[:, ActivityDescriptorProperties.start], 0)
        self.assertTrueAll(self.sleep_model.plan_dispatched)
        self.assertEqualAll(self.activity_manager.current_activity, self.sleep_model.sleep_activity.id)
        self.assertFalseAll(self.activity_manager.current_activity_interruptable)

        self.walk_engine(1)
        self.assertEqualAll(self.activity_manager.current_activity, self.sleep_model.sleep_activity.id)

    def test_default_controller_integration(self):
        self.setup_engine(sleep=False, default=True, use_office=True)

        office = self.w.universe.regions[2]
        work_descriptor = office.available_activities[0]
        coffee_break_descriptor = office.available_activities[1]

        self.location_manager.start_delay = lambda x: np.full(x, 0, dtype=int)

        self.walk_engine(10)
        self.w.relocator.move_agents(self.population.index, office)

        self.walk_engine(1)

        expected = [coffee_break_descriptor.activity_class.id] * len(self.population)
        self.assertEqualAll(self.activity_manager.current_activity, expected,
                            msg=f"{expected}, {self.activity_manager.current_activity}")

        self.walk_engine(coffee_break_descriptor.duration())
        expected = [work_descriptor.activity_class.id] * len(self.population)

        for i in range(coffee_break_descriptor.blocks_for()):
            self.assertEqualAll(self.activity_manager.current_activity, expected,
                                msg=f"{i}, {expected}, {self.activity_manager.current_activity}")

            self.walk_engine(1)

        expected = [coffee_break_descriptor.activity_class.id] * len(self.population)
        self.assertEqualAll(self.activity_manager.current_activity, expected,
                            msg=f"{expected}, {self.activity_manager.current_activity}")

        self.walk_engine(2)
        expected = self.activity_manager.activity_list.get_in_progress(self.population.index,
                                                                       work_descriptor.activity_class.id)
        self.assertFalseAll(expected.astype(bool),
                            msg=f"{expected}")

        # Side effect moves agents home
        self.w.assign_agents_to_worlds()
        self.assertEqualAll(self.activity_manager.current_activity, -1,
                            msg=f"{-1}, {self.activity_manager.current_activity}")

        self.assertEqualAll(self.activity_manager.current_descriptors, -1,
                            msg=f"{-1}, {self.activity_manager.current_descriptors}")

    def test_controlled_activities_availability(self):
        self.setup_engine(sleep=False, default=True, use_office=True)

        activity_id = Toilet().activity_class.id
        self.location_manager.plan[:] = Toilet(duration=3, blocks_for=60).create_specs().specifications
        self.location_manager.has_plan[:] = True
        for agent in self.population.index:
            ids_under_control = [act.activity_class.id for act in self.location_manager.descriptor_index[agent]]
            self.assertFalse(activity_id not in ids_under_control)

        self.location_manager.started_activities(activity_id, 0, self.population.index)

        for agent in self.population.index:
            ids_under_control = [act.activity_class.id for act in self.location_manager.descriptor_index[agent]]
            self.assertTrue(activity_id not in ids_under_control)

        self.location_manager.unblocked_activities(activity_id, 0, self.population.index)
        for agent in self.population.index:
            ids_under_control = [act.activity_class.id for act in self.location_manager.descriptor_index[agent]]
            self.assertFalse(activity_id not in ids_under_control)

    def test_delay_start(self):
        self.setup_engine(sleep=False, default=True, use_office=True)

        office = self.w.universe.regions[2]
        work_descriptor = office.available_activities[0]
        coffee_break_descriptor = office.available_activities[1]
        coffee_break = self.activity_manager.activity_list.activities[coffee_break_descriptor.activity_class.id]
        self.w.relocator.move_agents(self.population.index, office)

        self.location_manager.update_activity_plan(0)
        plan = self.location_manager.plan.copy()
        activity_started = np.full(len(self.population), -1)

        def start_callback(act_id, t, start_selector):
            if act_id == coffee_break_descriptor.activity_class.id:
                activity_started[start_selector] = t

        coffee_break.register_start_callbacks(start_callback)

        for i in range(max(self.location_manager.plan[:, ActivityDescriptorProperties.start])+2):
            print(self.location_manager.has_plan)
            self.walk_engine(1)

        self.assertEqualAll(plan[:, ActivityDescriptorProperties.start], activity_started,
                            msg=f"{activity_started}, {plan[:, ActivityDescriptorProperties.start]}")






