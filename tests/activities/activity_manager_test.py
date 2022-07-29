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
import os.path
from unittest import skip

import numpy as np

from i2mb.activities import ActivityProperties, TypesOfLocationBlocking, ActivityDescriptorProperties
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.atomic_activities import Sleep, Work
from i2mb.activities.base_activity import ActivityNone
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time
from i2mb.worlds import Apartment
from tests.i2mb_test_case import I2MBTestCase
from tests.world_tester import WorldBuilder


class TestActivityManager(I2MBTestCase):
    def setUp(self) -> None:
        self.population_size = 10
        self.population = AgentList(self.population_size)
        global_time.set_sim_time(0)

    def init_manager_and_test(self, relocator=None):
        activities = [c(self.population) for c in [Sleep, Work]]
        activity_manager = ActivityManager(self.population, relocator=relocator)
        for activity in activities:
            activity_manager.register_activity(activity)

        test_pattern = activities
        activity_manager.register_location_activities()
        return activity_manager, test_pattern

    def init_population_list_and_test(self):
        activity_list, test_pattern = self.init_manager_and_test()
        return activity_list, test_pattern, self.population

    def test_creation(self):
        activity_manager, test_pattern = self.init_manager_and_test()
        self.assertListEqual([type(c) for c in activity_manager.activity_list.activities],
                             [type(c) for c in [ActivityNone(self.population)] + test_pattern],  # noqa
                             msg=f"Activity list:\n{activity_manager}\nTestPattern:\n{test_pattern}")

        self.assertTrue((activity_manager.activity_list.activity_values[:, :, :] == 0).all())

        # test tha ActivityList values and ActivityPrimitive values are in sync
        activity_manager.activity_list.activity_values[np.ix_([4, 5, 6], [ActivityProperties.start], [0, 2])] = 3
        status_1 = activity_manager.activity_list.activities[0].get_start()
        test_1 = activity_manager.activity_list.activity_values[:, ActivityProperties.start, 0]

        status_2 = activity_manager.activity_list.activities[2].get_start()
        test_2 = activity_manager.activity_list.activity_values[:, ActivityProperties.start, 2]

        self.assertListEqual(status_1.tolist(), test_1.tolist())
        self.assertListEqual(status_2.tolist(), test_2.tolist())

    def test_stage_activities(self):
        activity_manager, test_pattern = self.init_manager_and_test()
        activity_specs = ActivityDescriptorSpecs(act_idx=1, start=0, duration=50, location_ix=-1)
        ids = [0, 1, 2, 3]
        activity_manager.stage_activity(activity_specs.specifications, ids)

        expected = activity_specs.specifications.tolist() * len(ids)
        staged = [list(act) for act in activity_manager.current_descriptors[ids]]

        self.assertListEqual(expected, staged)

        no_ids = list(range(4, len(self.population)))
        staged = activity_manager.current_descriptors[no_ids]
        self.assertTrue((staged == -1).all(), msg=f"{staged}")

    def test_stage_with_blocked_activities(self):
        activity_manager, test_pattern = self.init_manager_and_test()

        # Blocking Activity ID 1
        activity_manager.activity_list.activity_values[np.ix_([0, 1], [ActivityProperties.blocked_for], [1])] = 60

        activity_specs = ActivityDescriptorSpecs(act_idx=1, start=0, duration=50, location_ix=-1)
        ids = [0, 1, 2, 3]
        staged = activity_manager.stage_activity(activity_specs.specifications, ids)
        expected = [False, False, True, True]

        self.assertListEqual(expected, staged.tolist())

    def test_start(self):
        activity_list, test_pattern, population = self.init_population_list_and_test()

        # Test that un-staged activities raise ValueError
        self.assertRaises(ValueError, activity_list.start_activities, [5, 6])

        global_time.set_sim_time(5)
        ids = [5, 6]
        activity_specs = ActivityDescriptorSpecs(2, 0, 50, 0, 60, 0)
        activity_list.stage_activity(activity_specs.specifications, ids)
        activity_list.start_activities(ids)
        start_status = np.array(list(activity_list.activity_list.activity_values[:, ActivityProperties.start, 2]))
        in_progress = np.array(list(activity_list.activity_list.activity_values[:, ActivityProperties.in_progress, 2]))
        test_start = np.array([0] * self.population_size)
        test_start[ids] = 5
        self.assertTrue((start_status == test_start).all(),
                        msg=f"Start times:\n{start_status}\nTestPattern:\n{test_start}")

        start_status = np.array(list(activity_list.activity_list.activity_values[:, ActivityProperties.start, [0, 1]]))
        self.assertTrue((start_status == 0).all(),
                        msg=f"Start times:\n{start_status}\nTestPattern: == 0")

        test_in_progress = np.zeros_like(in_progress)
        test_in_progress[ids] = 1
        self.assertTrue((in_progress == test_in_progress).all(),
                        msg=f"In progress status:\n{in_progress}\nTestPattern:\n{test_in_progress}")

        # Test that descriptor is reset.
        self.assertTrue((activity_list.current_descriptors == -1).all(),
                        msg=f"Current Descriptors are not properly reset. {activity_list.current_descriptors}")

    def test_update_time(self):
        activity_manager, test_pattern, population = self.init_population_list_and_test()

        ids = [3, 5, 6]
        activity_specs = ActivityDescriptorSpecs(0, 0, 50, 0, 60, 0)
        activity_manager.stage_activity(activity_specs.specifications, ids)
        activity_manager.start_activities(ids)

        for i in range(5):
            global_time.set_sim_time(i)
            activity_manager.pre_step(i)
            activity_manager.step(i)
            activity_manager.post_step(i)

        elapsed_status = np.array(list(map(lambda x: x.get_elapsed(), activity_manager.activity_list.activities)))
        accumulated_status = np.array(list(map(lambda x: x.get_accumulated(), activity_manager.activity_list.activities)))
        test_start = np.zeros_like(elapsed_status)
        test_start[[0], ids] = 5
        self.assertTrue((elapsed_status == test_start).all(),
                        msg=f"Elapsed times:\n{elapsed_status}\nTestPattern:\n{test_start}")

        self.assertTrue((accumulated_status == test_start).all(),
                        msg=f"Accumulated times:\n{accumulated_status}\nTestPattern:\n{test_start}")

    def test_stop_activities_with_duration(self):
        activity_manager, test_pattern, population = self.init_population_list_and_test()

        ids = [3, 5, 6]
        activity_specs = ActivityDescriptorSpecs(0, 0, 50, 0, 0, 0)
        activity_manager.stage_activity(activity_specs.specifications, ids)
        activity_manager.start_activities(ids)

        for i in range(60):
            activity_manager.step(i)
            activity_manager.pre_step(i)
            activity_manager.step(i)
            activity_manager.post_step(i)

        elapsed_status = np.array(list(map(lambda x: x.get_elapsed(), activity_manager.activity_list.activities)))
        accumulated_status = np.array(list(map(lambda x: x.get_accumulated(), activity_manager.activity_list.activities)))
        in_progress = np.array(list(map(lambda x: x.get_in_progress(), activity_manager.activity_list.activities)))
        test_start = np.zeros_like(elapsed_status)
        test_start[[0], ids] = 50

        self.assertTrue((elapsed_status == 0).all(),
                        msg=f"Elapsed times:\n{elapsed_status}\nTestPattern: == 0")

        self.assertTrue((in_progress == 0).all(),
                        msg=f"Elapsed times:\n{in_progress}\nTestPattern: == 0")

        self.assertTrue((accumulated_status == test_start).all(),
                        msg=f"Accumulated times:\n{accumulated_status}\nTestPattern:\n{test_start}")

    def test_stop(self):
        activity_manager, test_pattern, population = self.init_population_list_and_test()

        ids = [3, 5, 6]
        activity_specs = ActivityDescriptorSpecs(0, 0, 50, 0, 60, 0)
        activity_manager.stage_activity(activity_specs.specifications, ids)
        activity_manager.start_activities(ids)
        for i in range(5):
            global_time.set_sim_time(i)
            activity_manager.pre_step(i)
            activity_manager.step(i)
            activity_manager.post_step(i)

        global_time.set_sim_time(i + 1)
        activity_manager.stop_activities(i + 1, ids)

        start_status = np.array(list(map(lambda x: x.get_start(), activity_manager.activity_list.activities)))
        in_progress = np.array(list(map(lambda x: x.get_in_progress(), activity_manager.activity_list.activities)))
        elapsed_status = np.array(list(map(lambda x: x.get_elapsed(), activity_manager.activity_list.activities)))
        accumulated_status = np.array(list(map(lambda x: x.get_accumulated(), activity_manager.activity_list.activities)))

        test_start = [0] * self.population_size
        self.assertTrue((start_status == test_start).all(),
                        msg=f"Start times:\n{start_status}\nTestPattern:\n{test_start}")

        test_start = [0] * self.population_size
        self.assertTrue((in_progress == test_start).all(),
                        msg=f"In progress status:\n{in_progress}\nTestPattern:\n{test_start}")

        test_start = [0] * self.population_size
        self.assertTrue((elapsed_status == test_start).all(),
                        msg=f"Elapsed times:\n{elapsed_status}\nTestPattern:\n{test_start}")

        test_start = np.zeros_like(accumulated_status)
        test_start[[0], ids] = 5
        self.assertTrue((accumulated_status == test_start).all(),
                        msg=f"Accumulated times:\n{accumulated_status}\nTestPattern:\n{test_start}")

    def test_start_with_relocation(self):
        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                             sim_duration=global_time.make_time(day=4), no_gui=True)

        relocator = world.relocator
        activity_manager, _ = self.init_manager_and_test(relocator)

        # Test that the agents do not move
        ids = [3, 5, 6]
        locations = self.population.location.copy()
        activity_specs = ActivityDescriptorSpecs(0, 0, 50, 0, 0, location_ix=0)
        activity_manager.stage_activity(activity_specs.specifications, ids)
        activity_manager.start_activities(ids)
        locations_after_start = self.population.location

        self.assertListEqual(locations.tolist(), locations_after_start.tolist())

        # Test correct in_progress status
        in_progress = np.array(list(activity_manager.activity_list.activity_values[:, ActivityProperties.in_progress, :]))
        test_start = np.zeros_like(in_progress)
        test_start[ids, 0] = 1
        self.assertTrue((in_progress == test_start).all(),
                        msg=f"In progress status:\n{in_progress}\nTestPattern:\n{test_start}")

        # Test that agents moved
        activity_specs = ActivityDescriptorSpecs(2, 0, 0, 0, 0, location_ix=world.universe.regions[0].regions[2].index)
        activity_manager.stage_activity(activity_specs.specifications, ids)
        activity_manager.start_activities(ids)
        expected_locations = locations.copy()
        expected_locations[ids] = world.universe.regions[0].regions[2]
        self.assertListEqual(self.population.location.tolist(), expected_locations.tolist())

        # Test correct in_progress status
        in_progress = np.array(list(activity_manager.activity_list.activity_values[:, ActivityProperties.in_progress, :]))
        test_start = np.zeros_like(in_progress)
        test_start[ids, 2] = 1
        self.assertTrue((in_progress == test_start).all(),
                        msg=f"In progress status:\n{in_progress}\nTestPattern:\n{test_start}")

    def test_region_unblocking(self):
        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                             sim_duration=global_time.make_time(day=4), no_gui=True)

        relocator = world.relocator
        activity_manager, _ = self.init_manager_and_test(relocator)


        # Block locations
        ids = [3, 5, 6]
        regions_idx = np.unique(np.searchsorted(world.universe.region_index[:, 0],
                                                [r.id for r in self.population.location[ids]]))
        world.universe.block_locations(regions_idx, True)
        test_pattern = world.universe.blocked_locations.copy()
        occupants_reg3 = world.universe.region_index[regions_idx[0], 2].population.index
        occupants_reg4 = world.universe.region_index[regions_idx[1], 2].population.index

        # Test that moving agents does not unblock the space if at least one agent remains
        relocator.move_agents(ids[:1],  world.universe.regions[2])
        self.assertListEqual(world.universe.blocked_locations.tolist(), test_pattern.tolist())

        # Unblock the first corridor by moving everyone out.
        relocator.move_agents(occupants_reg3, world.universe.regions[2])
        test_pattern[regions_idx[0]] = False
        self.assertListEqual(world.universe.blocked_locations.tolist(), test_pattern.tolist())

        # Test stopping activity of multiple agents should not unblock the space
        activity_manager.stop_activities(0, occupants_reg4)
        self.assertListEqual(world.universe.blocked_locations.tolist(), test_pattern.tolist())

        # Stopping activity, when there is only one agent remaining, unblocks the space
        relocator.move_agents(occupants_reg4[:-1], world.universe.regions[2])
        activity_manager.stop_activities(0, occupants_reg4)
        test_pattern[regions_idx[1]] = False
        self.assertListEqual(world.universe.blocked_locations.tolist(), test_pattern.tolist())

    def test_relocation_callback(self):
        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                             sim_duration=global_time.make_time(day=4), no_gui=True)

        relocator = world.relocator
        activity_list, _ = self.init_manager_and_test(relocator)
        activity_descriptor = ActivityDescriptorSpecs(2)
        activity_list.stage_activity(activity_descriptor.specifications, [2])
        activity_list.start_activities([2])

        # Run for a bit
        for i in range(5):
            global_time.set_sim_time(i)
            activity_list.pre_step(i)
            activity_list.step(i)
            activity_list.post_step(i)

        self.assertTrue(activity_list.current_activity[2] == 2)
        self.assertTrue((activity_list.current_descriptors[2] == -1).all(),
                        msg=f"{activity_list.current_descriptors[2]}")

        # Move 2 to new location, check that no activity is ongoing
        location = world.universe.regions[0].regions[3]
        relocator.move_agents([2], location)
        self.assertTrue(activity_list.current_activity[2] == -1,
                        msg=f"{activity_list.current_activity[2]}")

    def test_block_location_with_shared_blocking(self):
        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                             sim_duration=global_time.make_time(day=4), no_gui=True)

        relocator = world.relocator
        activity_list, _ = self.init_manager_and_test(relocator)
        location = world.universe.regions[0].regions[3]
        activity_specs = ActivityDescriptorSpecs(act_idx=1, start=0, duration=50, location_ix=location.index,
                                                 blocks_location=TypesOfLocationBlocking.shared)

        ids = [0, 1, 2, 3]
        activity_list.stage_activity(activity_specs.specifications, ids)

        expected = activity_specs.specifications.tolist() * len(ids)
        staged = [list(act) for act in activity_list.current_descriptors[ids]]

        self.assertListEqual(expected, staged)

        no_ids = list(range(4, len(self.population)))
        staged = activity_list.current_descriptors[no_ids]
        self.assertTrue((staged == -1).all(), msg=f"{staged}")

    def test_location_blocking_with_wait_blocking(self):
        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                             sim_duration=global_time.make_time(day=4), no_gui=True)

        relocator = world.relocator
        activity_list, _ = self.init_manager_and_test(relocator)
        location = world.universe.regions[0].regions[3]
        activity_descriptor = ActivityDescriptorSpecs(2, location_ix=location.index,
                                                      blocks_location=TypesOfLocationBlocking.wait)

        relocator.move_agents([1], location)
        activity_list.stage_activity(activity_descriptor.specifications, [2])
        activity_list.start_activities([2])

        # Test the activity is still in the stage area
        self.assertFalse((activity_list.current_descriptors[2] == -1).all())
        relocator.move_agents([1], location.parent)

        # Move the agent out of the way
        activity_list.start_activities([2])
        self.assertTrue((activity_list.current_descriptors[2] == -1).all())

        # Test the activity is started
        self.assertTrue(activity_list.current_activity[2] == 2)

    @skip("Not implemented yet")
    def test_block_location_with_rejecting_blocking(self):
        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                             sim_duration=global_time.make_time(day=4), no_gui=True)

        relocator = world.relocator
        activity_list, _ = self.init_manager_and_test(relocator)
        location = world.universe.regions[0].regions[3]
        activity_descriptor = ActivityDescriptorSpecs(2, location_ix=location.index,
                                                      blocks_location=TypesOfLocationBlocking.rejecting)

        # Move every one to location
        relocator.move_agents([0, 1, 2, 3, 4], location)

        # Block and stage activity
        activity_list.stage_activity(activity_descriptor.specifications, [2])
        activity_list.start_activities([2])

        # Test the activity is started
        self.assertTrue(activity_list.current_activity[2] == 2)

        # Test that the activity is in location
        self.assertTrue(self.population[2].location == location)

        # Test that everyone else is in the adjacent space.
        adjacent_space = world.universe.regions[0].regions[2]
        self.assertTrue(self.population[[0, 1, 3, 4]].location == adjacent_space)

    @skip("Not implemented yet")
    def test_block_parent(self):
        self.assertTrue(False)

    def test_activity_diary(self):
        test_history_file = "/tmp/i2mb_test_suite_activity_history.csv"
        if os.path.exists(test_history_file):
            os.remove(test_history_file)

        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                             sim_duration=global_time.make_time(day=4), no_gui=True)

        relocator = world.relocator
        activity_manager, _ = self.init_manager_and_test(relocator)
        activity_manager.write_diary = True
        activity_manager.post_init("/tmp/i2mb_test_suite")


        self.assertTrue(os.path.exists(test_history_file),
                        msg="Test history file does not exist")

        location = world.universe.regions[0].regions[3]
        durations = [10, 15, 20, 25]
        activity_descriptor = ActivityDescriptorSpecs(2, duration=10, location_ix=location.index,
                                                      blocks_location=TypesOfLocationBlocking.rejecting, size=4)

        activity_descriptor.specifications[:, ActivityDescriptorProperties.duration] = durations

        activity_manager.stage_activity(activity_descriptor.specifications, [0, 1, 2, 3])
        activity_manager.start_staged_activities()

        for i in range(30):
            global_time.set_sim_time(i)
            activity_manager.pre_step(i)
            activity_manager.step(i)
            activity_manager.post_step(i)

        # Trigger file closing
        del activity_manager

        with open(test_history_file) as thf:
            for lix, line in enumerate(thf.readlines()):
                if lix == 0:
                    self.assertListEqual(ActivityManager.file_headers, line.split(","))
                    continue

                test_line = [f"{lix - 1}", "Work", "0", f"{durations[lix -1]}", type(location).__name__]
                self.assertListEqual(test_line, line.split(","))

    def test_activity_descriptors_match_location(self):
        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                             sim_duration=global_time.make_time(day=4), no_gui=True)

        relocator = world.relocator
        activity_list, _ = self.init_manager_and_test(relocator)

        for r in world.universe.list_all_regions():
            for act_descriptor in r.local_activities:
                act_type = act_descriptor.activity_class
                self.assertNotEqual(act_type.id, -1)

    @skip("Not implemented")
    def test_starting_with_mixed_none_and_specific_location(self):
        # Current relocation test does this:  ids[relocated_ids | ~update_current | same_location]
        self.fail("You need to implement this test")

    @skip("Not implemented")
    def test_starting_with_mixed_specific_location_and_current_location(self):
        # Current relocation test does this: ids[relocated_ids | ~update_current | same_location]
        self.fail("You need to implement this test")

    def test_triggered_location_action(self):
        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                             sim_duration=global_time.make_time(day=4), no_gui=True)

        relocator = world.relocator
        activity_manager, _ = self.init_manager_and_test(relocator)

        called_action = [False]

        def location_action(region):
            called_action[0] = True
            return [], []

        activity_manager.location_activity_handler.register_handler_action(world.universe.regions[0].index,
                                                                           location_action)

        for i in range(5):
            global_time.set_sim_time(i)
            activity_manager.pre_step(i)
            activity_manager.step(i)
            activity_manager.post_step(i)
            self.assertTrue(called_action[0], msg="Location action not called")



