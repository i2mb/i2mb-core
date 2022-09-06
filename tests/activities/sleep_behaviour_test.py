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
import random
from functools import partial
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from i2mb.activities import ActivityDescriptorProperties
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.atomic_activities import Sleep
from i2mb.activities.controllers.sleep_controller import SleepBehaviourController
from i2mb.utils import global_time, time
from i2mb.worlds import Apartment
from tests.i2mb_test_case import I2MBTestCase
from tests.world_tester import WorldBuilder

global_time.ticks_hour = 60 // 5


class TestSleepBehaviour(I2MBTestCase):
    def setup_engine(self, callbacks=None, no_gui=True):
        self.w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=5), sim_duration=global_time.make_time(day=3),
                              update_callback=callbacks, no_gui=no_gui)

        self.population = self.w.population
        self.activity_manager = ActivityManager(self.w.population, self.w.relocator)

        # sleep_duration = partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(hour=1))
        # sleep_midpoint = partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(hour=1))
        def sleep_duration(shape):
            return np.ones(shape, dtype=int) * global_time.make_time(hour=8)

        def sleep_midpoint(shape):
            return np.ones(shape, dtype=int) * global_time.make_time(hour=1)

        sleep_model = SleepBehaviourController(self.w.population, self.activity_manager,
                                               sleep_duration,
                                               sleep_midpoint)
        self.activity_manager.register_activity_controller(sleep_model)
        self.sleep_model = sleep_model
        self.w.engine.models.extend([sleep_model, self.activity_manager])
        self.w.engine.post_init_modules()
        self.engine_iterator = self.w.engine.step()

    def setup_random_sleep_scheduler(self):
        sleep_duration = partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(minutes=30))
        sleep_midpoint = partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(minutes=30))
        self.sleep_model.sleep_midpoint = sleep_midpoint
        self.sleep_model.sleep_duration = sleep_duration


class TestSleepBehaviourGui(TestSleepBehaviour):
    def test_sleep_in_apartment(self):
        def move_agents_to_rooms(world_builder, frame):
            delta = global_time.make_time(hour=12)
            if frame == global_time.to_current(delta, frame):
                for apartment in world_builder.worlds:
                    if type(apartment) is not Apartment:
                        continue

                    population = apartment.inhabitants
                    world_builder.relocator.move_agents(population.index, apartment.living_room)

            self.assertFalse(np.isnan(self.population.position).any())

        self.setup_engine(move_agents_to_rooms, no_gui=False)
        plt.show()

    def test_sleep_in_apartment_random_sleep_schedule(self):
        def move_agents_to_rooms(world_builder, frame):
            delta = global_time.make_time(hour=12)
            if frame == global_time.to_current(delta, frame):
                for apartment in world_builder.worlds:
                    if type(apartment) is not Apartment:
                        continue

                    population = apartment.inhabitants
                    world_builder.relocator.move_agents(population.index, apartment.living_room)

            self.assertFalse(np.isnan(self.population.position).any())

        self.setup_engine(move_agents_to_rooms, no_gui=False)
        self.setup_random_sleep_scheduler()
        plt.show()


class TestSleepBehaviourNoGui(TestSleepBehaviour):
    def walk_engine(self, num_steps):
        t = self.w.engine.time
        for frame, rs in enumerate(self.engine_iterator, start=t):
            stop = self.w.process_stop_criteria(frame)
            if stop:
                return

            self.w.update_callback(self.w, frame)

            num_steps -= 1
            if num_steps <= 0:
                return

    def test_setup(self):
        self.setup_engine(no_gui=True)
        self.assertEqual(self.sleep_model.sleep_activity.id, Sleep.id)
        self.assertEqual(self.sleep_model.sleep_activity, self.activity_manager.activity_list.activities[Sleep.id])

    def test_sleep_in_apartment_no_gui_simple_schedule(self):
        def move_agents_to_rooms(world_builder, frame):
            delta = global_time.make_time(hour=12)
            if frame == global_time.to_current(delta, frame):
                for apartment in world_builder.worlds:
                    if type(apartment) is not Apartment:
                        continue

                    population = apartment.inhabitants
                    world_builder.relocator.move_agents(population.index, apartment.living_room)

            __expected_current_activity = [-1] * len(self.w.population)
            if (self.activity_manager.current_activity == Sleep.id).all():
                self.assertTrue(self.activity_manager.activity_list.activities[Sleep.id].in_bed.all(),
                                msg=f"Error occurred at frame {frame}")
                __expected_current_activity = [Sleep.id] * len(self.w.population)
                expected_location = list(self.sleep_model.sleep_profiles.specifications[:, 5])
                current_location = list([r.id for r in self.population.location])
                self.assertListEqual(expected_location, current_location,
                                     msg=f"Error occurred at frame {frame}")
            elif global_time.hour(frame) < 12:
                expected_location = list(self.sleep_model.sleep_profiles.specifications[:, 5])
                current_location = list([r.id for r in self.population.location])
                self.assertListEqual(expected_location, current_location,
                                     msg=f"Error occurred at frame {frame}")

            __current_activity = list(self.activity_manager.current_activity)
            self.assertListEqual(__expected_current_activity, __current_activity,
                                 msg=f"Error occurred at frame {frame}")

            self.assertFalse((self.population.position == np.nan).any())

        self.setup_engine(move_agents_to_rooms, no_gui=True)
        self.walk_engine(1)
        expected_current_activity = [Sleep.id] * len(self.w.population)
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_current_activity, current_activity)

        self.walk_engine(global_time.make_time(day=4))

    def test_sleep_walkers_nogui(self):
        def move_agents_to_rooms(world_builder, frame):
            delta = global_time.make_time(hour=4)
            if frame == global_time.to_current(delta, frame):
                for apartment in world_builder.worlds:
                    if type(apartment) is not Apartment:
                        continue

                    population = apartment.inhabitants
                    world_builder.relocator.move_agents(population.index, apartment.living_room)

                return

            __expected_current_activity = [-1] * len(self.w.population)
            if (self.activity_manager.current_activity == Sleep.id).all():
                self.assertTrue(self.activity_manager.activity_list.activities[Sleep.id].in_bed.all(),
                                msg=f"Error occurred at frame {frame}")
                __expected_current_activity = [Sleep.id] * len(self.w.population)
                expected_location = list(self.sleep_model.sleep_profiles.specifications[:, 5])
                current_location = list([r.id for r in self.population.location])
                self.assertListEqual(expected_location, current_location,
                                     msg=f"Error occurred at frame {frame}")

                self.assertTrue(self.population.sleep.all())

            __current_activity = list(self.activity_manager.current_activity)
            self.assertListEqual(__expected_current_activity, __current_activity,
                                 msg=f"Error occurred at frame {frame}")

            self.assertFalse((self.population.position == np.nan).any())

        self.setup_engine(move_agents_to_rooms, no_gui=True)
        self.walk_engine(1)
        expected_current_activity = [Sleep.id] * len(self.w.population)
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_current_activity, current_activity)

        self.walk_engine(global_time.make_time(day=4))

    def test_random_sleep_patterns(self):
        def move_agents_to_rooms(world_builder, frame):
            delta = global_time.make_time(hour=12)
            if frame == global_time.to_current(delta, frame):
                for apartment in world_builder.worlds:
                    if type(apartment) is not Apartment:
                        continue

                    population = apartment.inhabitants
                    world_builder.relocator.move_agents(population.index, apartment.living_room)

                return

            if (self.activity_manager.current_activity == Sleep.id).any():
                expected = (self.activity_manager.current_activity == Sleep.id).ravel()
                current = self.activity_manager.activity_list.activities[Sleep.id].sleep.ravel()
                in_progress = self.activity_manager.activity_list.activities[Sleep.id].get_in_progress()
                self.assertListEqual(expected.tolist(), current.tolist(),
                                 msg=f"Error occurred at frame {frame}: {current}, {expected}, {in_progress}")

                sleeping = self.activity_manager.current_activity == Sleep.id
                expected_location = list(self.sleep_model
                                         .sleep_profiles
                                         .specifications[sleeping, ActivityDescriptorProperties.location_ix])
                current_location = list([r.id for r in self.population.location[sleeping]])
                self.assertListEqual(expected_location, current_location,
                                     msg=f"Error occurred at frame {frame}")

                self.assertEqual((self.activity_manager.current_activity == Sleep.id).sum(),
                                 self.population.sleep.sum(),
                                 msg=f"Error occurred at frame {frame}")

            self.assertFalse((self.population.position == np.nan).any())

        self.setup_engine(move_agents_to_rooms, no_gui=True)
        self.setup_random_sleep_scheduler()
        self.walk_engine(1)
        expected_current_activity = [Sleep.id] * len(self.w.population)
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_current_activity, current_activity)

        self.walk_engine(global_time.make_time(day=4))

    def test_callback_registration(self):
        self.setup_engine(no_gui=True)
        self.assertListEqual(self.sleep_model.sleep_activity._ActivityPrimitive__stop_callback[-1:],
                             [self.sleep_model.reset_sleep_on_stop])

    def test_sleep_unblocking(self):
        self.setup_engine(no_gui=True)
        self.setup_random_sleep_scheduler()
        while True:
            self.walk_engine(1)
            sleeping = self.sleep_model.sleep_activity.sleep.ravel()
            if sleeping.any():
                interrupt = self.population.index[sleeping][0]
                break

        self.w.relocator.move_agents([interrupt], self.w.universe.regions[2])
        for i in range(self.sleep_model.minimum_up_time):
            blocked_for = self.sleep_model.sleep_activity.get_blocked_for()[interrupt]
            expected_blocked_for = self.sleep_model.minimum_up_time - i
            self.assertEqual(blocked_for, expected_blocked_for)
            self.walk_engine(1)

    def test_sleep_interruption(self):
        self.setup_engine(no_gui=True)
        self.setup_random_sleep_scheduler()
        self.walk_engine(1)
        expected_current_activity = [self.sleep_model.sleep_activity.id] * len(self.w.population)
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_current_activity, current_activity)
        interrupt = -1
        while True:
            self.walk_engine(1)
            sleeping = self.sleep_model.sleep_activity.sleep.ravel()
            if sleeping.any():
                duration = self.sleep_model.sleep_activity.get_duration()[sleeping]
                elapsed = self.sleep_model.sleep_activity.get_elapsed()[sleeping]
                candidates = (duration / 4 < elapsed).ravel()
                idx = self.population.index[sleeping]

                if not len(idx[candidates]):
                    continue

                interrupt = random.choice(idx[candidates])
                self.w.relocator.move_agents([interrupt], self.w.universe.regions[2])
                break

        expected_current_activity = [self.sleep_model.sleep_activity.id] * len(self.w.population)
        expected_current_activity[interrupt] = -1
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_current_activity, current_activity)

        self.walk_engine(self.sleep_model.minimum_up_time // 2)
        self.w.relocator.move_agents([interrupt], self.population.home[interrupt])

        # Generate the plan for the interrupted agent
        self.sleep_model.update_sleep_schedule(time())

        # Put starting time in the past
        start = time() - 2
        self.sleep_model.sleep_profiles.specifications[interrupt, ActivityDescriptorProperties.start] = start

        self.assertFalse(self.sleep_model.plan_dispatched[interrupt])
        self.assertGreater(self.sleep_model.sleep_activity.get_blocked_for()[interrupt], 0)

        self.walk_engine(global_time.make_time(day=3))
        self.assertNotEqual(self.sleep_model.sleep_profiles.specifications[interrupt,
                                                                           ActivityDescriptorProperties.start],
                            start)










