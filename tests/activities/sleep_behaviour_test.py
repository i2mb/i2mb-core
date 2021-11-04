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
from functools import partial
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.base_activity import ActivityList
from i2mb.activities.controllers.sleep import SleepBehaviourController
from i2mb.utils import global_time
from i2mb.worlds import BedRoom, Apartment
from tests.world_tester import WorldBuilder

global_time.ticks_hour = 60 // 5


class TestSleepBehaviour(TestCase):
    def setup_engine(self, callbacks=None, no_gui=True):
        self.w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=5), sim_duration=global_time.make_time(day=3),
                              update_callback=callbacks, no_gui=no_gui)
        print(f"Running for {global_time.make_time(day=3)} ticks")

        self.population = self.w.population
        self.activity_list = ActivityList(self.w.population)
        self.activity_manager = ActivityManager(self.w.population, self.w.universe, activities=self.activity_list)

        # sleep_duration = partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(hour=1))
        # sleep_midpoint = partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(hour=1))
        def sleep_duration(shape):
            return np.ones(shape, dtype=int) * global_time.make_time(hour=8)

        def sleep_midpoint(shape):
            return np.ones(shape, dtype=int) * global_time.make_time(hour=1)

        sleep_model = SleepBehaviourController(self.w.population, self.activity_manager, self.w.universe,
                                               sleep_duration,
                                               sleep_midpoint)
        self.sleep_model = sleep_model
        self.w.engine.models.extend([sleep_model, self.activity_manager])
        self.w.engine.post_init_modules()

    def walk_engine(self, num_steps):
        t = self.w.engine.time
        for frame, rs in enumerate(self.w.engine.step(), start=t):
            stop = self.w.process_stop_criteria(frame)
            if stop:
                return

            num_steps -= 1
            if num_steps <= 0:
                return

            self.w.update_callback(self.w, frame)

    def test_sleep_in_bedroom(self):
        w = WorldBuilder(BedRoom, world_kwargs=dict(num_beds=2), sim_duration=global_time.make_time(day=4))
        start = 0
        end = 5
        for w_ in w.worlds:
            w_.assign_beds(w.population.index[start:end])
            start += 5
            end += 5

        sleep_duration = partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(hour=1))
        sleep_midpoint = partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(hour=1))
        activity_list = ActivityList(w.population)

        sleep_model = SleepBehaviourController(w.population, activity_list, sleep_duration, sleep_midpoint)
        w.engine.models.append(sleep_model)
        plt.show()

    def test_sleep_in_apartment(self):
        def move_agents_to_rooms(world_builder, frame):
            delta = global_time.make_time(hour=12)
            if frame == global_time.to_current(delta, frame):
                for apartment in world_builder.worlds:
                    if type(apartment) is not Apartment:
                        continue

                    population = apartment.inhabitants
                    world_builder.universe.move_agents(population.index, apartment.living_room)

            self.assertFalse(np.isnan(self.population.position).any())

        self.setup_engine(move_agents_to_rooms, no_gui=False)
        plt.show()

    def test_sleep_in_apartment_no_gui_simple_schedule(self):
        def move_agents_to_rooms(world_builder, frame):
            delta = global_time.make_time(hour=12)
            if frame == global_time.to_current(delta, frame):
                for apartment in world_builder.worlds:
                    if type(apartment) is not Apartment:
                        continue

                    population = apartment.inhabitants
                    world_builder.universe.move_agents(population.index, apartment.living_room)

            self.assertTrue((self.activity_manager.interrupted_activities.num_items == 0).all(),
                            msg=f"Frame {frame}, \n{self.activity_manager.interrupted_activities.queue[:,:,0]}")
            self.assertTrue((self.activity_manager.planned_activities.num_items == 0).all())
            self.assertTrue((self.activity_manager.postponed_activities.num_items == 0).all())
            self.assertTrue((self.activity_manager.triggered_activities.num_items == 0).all())

            __expected_current_activity = [0] * len(self.w.population)
            if (self.activity_manager.current_activity == 1).all():
                self.assertTrue(self.activity_manager.activities.activities[1].in_bed.all(),
                                 msg=f"Error occurred at frame {frame}")
                __expected_current_activity = [1] * len(self.w.population)
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
        expected_current_activity = [1] * len(self.w.population)
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
                    world_builder.universe.move_agents(population.index, apartment.living_room)

                return

            self.assertTrue((self.activity_manager.interrupted_activities.num_items == 0).all(),
                            msg=f"Frame {frame}, \n{self.activity_manager.interrupted_activities.queue[:,:,0]}")
            self.assertTrue((self.activity_manager.planned_activities.num_items == 0).all())
            self.assertTrue((self.activity_manager.postponed_activities.num_items == 0).all())
            self.assertTrue((self.activity_manager.triggered_activities.num_items == 0).all())

            __expected_current_activity = [0] * len(self.w.population)
            if (self.activity_manager.current_activity == 1).all():
                self.assertTrue(self.activity_manager.activities.activities[1].in_bed.all(),
                                 msg=f"Error occurred at frame {frame}")
                __expected_current_activity = [1] * len(self.w.population)
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
        expected_current_activity = [1] * len(self.w.population)
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_current_activity, current_activity)

        self.walk_engine(global_time.make_time(day=4))
