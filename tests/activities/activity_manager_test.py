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

import matplotlib.pyplot as plt
import numpy as np

import i2mb.activities.activity_descriptors
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.atomic_activities import Sleep
from i2mb.activities.base_activity import ActivityList
from i2mb.activities.controllers.location_activities import LocationActivities
from i2mb.activities.controllers.sleep import SleepBehaviour
from i2mb.utils import global_time
from i2mb.worlds import Apartment
from tests.world_tester import WorldBuilder

global_time.ticks_scalar = 60 * 24


class TestActivityManager(TestCase):
    def test_activity_assignment_in_apartments(self):
        w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=6), sim_duration=global_time.make_time(day=4))
        activity_manager = ActivityManager(w.population, w.universe)
        local_activities = activity_manager.get_local_activities()
        self.assertListEqual(list(local_activities), [])

        for a in w.worlds:
            w.universe.move_agents(a.inhabitants.index, a.bedrooms[0])

        local_activities = activity_manager.get_local_activities()
        self.assertListEqual(list(set(type(c) for c in local_activities)), list({
                                                                                    i2mb.activities.activity_descriptors.Sleep,
                                                                                    i2mb.activities.activity_descriptors.Sleep}))

        for a in w.worlds:
            w.universe.move_agents(a.inhabitants.index, a.bathroom)

        local_activities = activity_manager.get_local_activities()
        self.assertListEqual(list(set(type(c) for c in local_activities)), list({
                                                                                    i2mb.activities.activity_descriptors.Toilet,
                                                                                    i2mb.activities.activity_descriptors.Sink,
                                                                                    i2mb.activities.activity_descriptors.Shower}))

        available_activities = activity_manager.get_activities_available()
        self.assertListEqual(list(set([type(c) for c in available_activities[0]])),
                             list({i2mb.activities.activity_descriptors.Rest,
                                   i2mb.activities.activity_descriptors.Toilet,
                                   i2mb.activities.activity_descriptors.Sink,
                                   i2mb.activities.activity_descriptors.Shower,
                                   i2mb.activities.activity_descriptors.Cook, i2mb.activities.activity_descriptors.Eat,
                                   i2mb.activities.activity_descriptors.Sleep}))

    def test_activity_manager_in_apartments(self):
        w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=6), sim_duration=global_time.make_time(day=4))

        activity_list = ActivityList(w.population)
        activity_manager = ActivityManager(w.population, w.universe, activities=activity_list)

        # Lets add the sleep routine
        sleep_distributions = dict(
            # Distributions for sleep duration and mid points
            sleep_duration=partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(hour=1)),
            sleep_midpoint=partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(hour=1))
        )

        sleep_module = SleepBehaviour(w.population, activity_list=activity_list, **sleep_distributions)

        w.engine.models.extend([sleep_module, activity_manager])
        w.engine.post_init_modules()
        plt.show(block=True)

    def test_activity_location_sleep_in_apartments(self):
        w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=6), sim_duration=global_time.make_time(day=10))

        activity_list = ActivityList(w.population)
        activity_manager = ActivityManager(w.population, w.universe, activities=activity_list)

        # Lets add the sleep routine
        sleep_distributions = dict(
            # Distributions for sleep duration and mid points
            sleep_duration=partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(hour=1)),
            sleep_midpoint=partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(hour=1))
        )

        sleep_module = SleepBehaviour(w.population, activity_list=activity_list, **sleep_distributions)

        local_activities = LocationActivities(w.population, w.universe, activities=activity_list)

        w.engine.models.extend([sleep_module, local_activities, activity_manager])
        w.engine.post_init_modules()
        plt.show(block=True)

    def test_activity_location_in_apartment(self):
        w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=6), sim_duration=global_time.make_time(day=4))

        activity_list = ActivityList(w.population)
        activity_manager = ActivityManager(w.population, w.universe, activities=activity_list)
        local_activities = LocationActivities(w.population, w.universe, activities=activity_list)

        w.engine.models.extend([local_activities, activity_manager])
        w.engine.post_init_modules()
        plt.show(block=True)

    def test_activity_location_sleep_in_apartments(self):
        def update_frame(world_builder, frame):
            hour = global_time.hour(frame)
            if hour == 9:
                world_builder.universe.move_agents(slice(None), world_builder.worlds[2])

            sleepers = activity_manager.current_activity == activity_manager.activities.activity_types.index(Sleep)
            # if sleepers.any():
            #     assert world_builder.population.in_bed[sleepers].all()

            if hour == 2:
                for home in world_builder.worlds[:-1]:
                    world_builder.universe.move_agents(home.inhabitants.index, home)

        w = WorldBuilder(Apartment, world_kwargs=dict(num_residents=6), sim_duration=global_time.make_time(day=10),
                         update_callback=update_frame)

        activity_list = ActivityList(w.population)
        activity_manager = ActivityManager(w.population, w.universe, activities=activity_list)

        def return_constant(constant, size):
            return np.full(size, constant)

        # Lets add the sleep routine
        sleep_distributions = dict(
            # Distributions for sleep duration and mid points
            sleep_duration=partial(return_constant, global_time.make_time(hour=8)),
            sleep_midpoint=partial(return_constant, global_time.make_time(hour=2))
        )

        sleep_module = SleepBehaviour(w.population, activity_list=activity_list, **sleep_distributions)

        local_activities = LocationActivities(w.population, w.universe, activities=activity_list)

        w.engine.models.extend([sleep_module, local_activities, activity_manager])
        w.engine.post_init_modules()
        plt.show(block=True)

