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
from i2mb.activities.controllers.sleep import SleepBehaviour
from i2mb.utils import global_time
from i2mb.worlds import BedRoom, Apartment
from tests.world_tester import WorldBuilder

global_time.ticks_scalar = 60 * 24


class TestSleepBehaviour(TestCase):
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

        sleep_model = SleepBehaviour(w.population, activity_list, sleep_duration, sleep_midpoint)
        w.engine.models.append(sleep_model)
        plt.show()

    def test_sleep_in_apartment(self):
        def move_agents_to_rooms(world_builder, frame):
            delta = global_time.make_time(hour=12)
            if frame == global_time.to_current(delta, frame):
                for apartment in world_builder.worlds:
                    population = apartment.inhabitants
                    world_builder.universe.move_agents(population.index, apartment.living_room)

        w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=5), sim_duration=global_time.make_time(day=3),
                         update_callback=move_agents_to_rooms)

        activity_list = ActivityList(w.population)
        activity_manager = ActivityManager(w.population, w.universe, activities=activity_list)

        sleep_duration = partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(hour=1))
        sleep_midpoint = partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(hour=1))

        sleep_model = SleepBehaviour(w.population, activity_list, sleep_duration, sleep_midpoint)
        w.engine.models.extend([sleep_model, activity_manager])
        w.engine.post_init_modules()
        plt.show()

    def test_sleep_in_apartment_no_gui(self):
        def move_agents_to_rooms(world_builder, frame):
            delta = global_time.make_time(hour=12)
            if frame == global_time.to_current(delta, frame):
                for apartment in world_builder.worlds:
                    population = apartment.inhabitants
                    world_builder.universe.move_agents(population.index, apartment.living_room)

        w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=5), sim_duration=global_time.make_time(day=3),
                         update_callback=move_agents_to_rooms, no_gui=True)


        print(f"Running for {global_time.make_time(day=3)} ticks")
        activity_list = ActivityList(w.population)
        activity_manager = ActivityManager(w.population, w.universe, activities=activity_list)

        sleep_duration = partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(hour=1))
        sleep_midpoint = partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(hour=1))

        sleep_model = SleepBehaviour(w.population, activity_list, sleep_duration, sleep_midpoint)
        w.engine.models.extend([sleep_model, activity_manager])
        w.engine.post_init_modules()
        for frame, rs in enumerate(w.engine.step()):
            stop = w.process_stop_criteria(frame)
            if stop:
                break

            elapsed_ix = activity_manager.activities.start_ix
            print(frame, activity_manager.activities.get_current_activity_property(elapsed_ix))


