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
from unittest import TestCase
from warnings import warn

import numpy as np

import i2mb.activities.atomic_activities as aa
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.base_activity import ActivityList
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time
from i2mb.worlds import Apartment
from tests.world_tester import WorldBuilder

global_time.ticks_hour = 12  # 5 minutes


class TestActivityManager(TestCase):
    def setUp(self) -> None:
        self.population = AgentList(10)
        self.activity_classes = [aa.Work, aa.Sink, aa.Shower, aa.Cook, aa.Rest]
        self.t = 0

    def setup_activity_manager(self):
        self.t = 0
        self.activity_list = ActivityList(self.population)
        for act_cls in self.activity_classes:
            self.activity_list.register(act_cls(self.population))

        self.activity_manager = ActivityManager(self.population, activities=self.activity_list)

    def setup_world(self):
        self.w = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                              sim_duration=global_time.make_time(day=4), no_gui=True)

    def walk_manager(self, num_steps):
        for step in range(num_steps):
            self.activity_manager.step(self.t)
            self.t += 1

    def programmed_activity_queue(self, ids=None):
        self.setup_activity_manager()
        if ids is None:
            ids = slice(None)

        activity_specs = ActivityDescriptorSpecs(3, 5, 50, 0, 0, 0)
        self.activity_manager.planned_activities[ids].append(activity_specs)
        # check the activity is not popped before 5 steps:
        self.walk_manager(4)
        self.assertTrue((self.activity_list.activity_values[ids, :, 1:] == 0).all(),
                        msg=f"{self.activity_list.activity_values}")
        # test popping queue
        self.walk_manager(4)
        self.assertTrue((self.activity_list.activity_values[ids, :, 3] == [5, 50, 2, 2, 1, 0]).all(),
                        msg=f"{self.activity_list.activity_values}")
        self.assertTrue((self.activity_list.activity_values[ids, :, 0] == [0, 0, 0, 6, 0, 0]).all(),
                        msg=f"{self.activity_list.activity_values}")

    def test_consuming_programmed_activities(self):
        self.setup_activity_manager()
        ids = slice(3, 6)
        # Discard mode
        for act_id in range(2, 5):
            activity_specs = ActivityDescriptorSpecs(act_id, 0, 0, 0, 0, 0)
            self.activity_manager.planned_activities[ids].append(activity_specs)

        for i in range(2, 10):
            self.walk_manager(1)
            if i > 4:
                i = 4

            self.assertEqual([i] * 3, list(self.activity_manager.current_activity[ids]))
            self.assertListEqual([0] * 10, list(self.activity_manager.interrupted_activities.num_items))

        # Interruptable
        for act_id in range(2, 5):
            activity_specs = ActivityDescriptorSpecs(act_id, 0, 15, 0, 0, 0)
            self.activity_manager.planned_activities[ids].append(activity_specs)

        for i in range(2, 10):
            self.walk_manager(1)
            if i > 4:
                i = 4

            self.assertEqual([i] * 3, list(self.activity_manager.current_activity[ids]))
            expected = [0] * 3 + [i-2] * 3 + [0] * 4
            self.assertListEqual(expected, list(self.activity_manager.interrupted_activities.num_items))

        # Check that the interrupted  queue waits for activities to finish before proceeding
        for i in range(4, 1, -1):
            if i == 4:
                self.walk_manager(1)
            else:
                self.walk_manager(14)

            self.assertEqual([i] * 3, list(self.activity_manager.current_activity[ids]))
            expected = [0] * 3 + [i - 2] * 3 + [0] * 4
            self.assertListEqual(expected, list(self.activity_manager.interrupted_activities.num_items))

        # Ensure default activity is active
        self.walk_manager(14)

        # Un interruptable
        for act_id in range(2, 5):
            activity_specs = ActivityDescriptorSpecs(act_id, 0, 15, block_for=1)
            self.activity_manager.planned_activities[ids].append(activity_specs)

        for i in range(45):
            self.walk_manager(1)
            self.assertEqual([i // 15 + 2] * 3, list(self.activity_manager.current_activity[ids]),
                             msg=f"Failed at {i}")
            self.assertListEqual([0] * 10, list(self.activity_manager.interrupted_activities.num_items))

    def test_current_activity_update(self):
        self.setup_activity_manager()
        self.walk_manager(4)
        elapsed_ix = self.activity_list.elapsed_ix
        accumulated_ix = self.activity_list.accumulated_ix
        self.assertTrue((self.activity_list.activity_values[:, [elapsed_ix, accumulated_ix], 0] == 4).all(),
                        msg=f"{self.activity_list.activity_values[:, [elapsed_ix, accumulated_ix], 0]}")

        self.assertTrue((self.activity_list.activity_values[:, [elapsed_ix, accumulated_ix], 1:] == 0).all(),
                        msg=f"{self.activity_list.activity_values[:, [elapsed_ix, accumulated_ix], 1:]}")

        in_progress_ix = self.activity_list.in_progress_ix
        self.assertTrue((self.activity_list.activity_values[:, [in_progress_ix], 0] == 1).all(),
                        msg=f"{self.activity_list.activity_values[:, [elapsed_ix, accumulated_ix, in_progress_ix], 0]}")

        # Test update of blocking_for values
        blocked_for_ix = self.activity_list.blocked_for_ix
        self.activity_list.activity_values[:, blocked_for_ix, 4] = 60
        self.walk_manager(10)
        self.assertTrue((self.activity_list.activity_values[:, [blocked_for_ix], 4] == 50).all(),
                        msg=f"{self.activity_list.activity_values[:, [elapsed_ix, accumulated_ix, blocked_for_ix], 4]}")

        self.walk_manager(60)
        self.assertTrue((self.activity_list.activity_values[:, [blocked_for_ix], 4] == 0).all(),
                        msg=f"{self.activity_list.activity_values[:, [elapsed_ix, accumulated_ix, blocked_for_ix], 4]}")

    def test_programmed_activities(self):
        self.programmed_activity_queue()

    def test_programmed_activities_view(self):
        self.programmed_activity_queue(slice(2, 4))

    def test_programmed_activities_view_ids(self):
        self.programmed_activity_queue([2, 4])

    def test_blocked_activity(self):
        self.setup_activity_manager()
        activity_specs = ActivityDescriptorSpecs(1, 0, 50, 0, 60, 0)
        self.activity_manager.planned_activities.append(activity_specs)
        self.walk_manager(30)

        blocked_for_ix = self.activity_list.blocked_for_ix
        self.assertTrue((self.activity_list.activity_values[:, [blocked_for_ix], 1] == 60).all(),
                        msg=f"{self.activity_list.activity_values[:, [blocked_for_ix], 1]}")

        self.activity_manager.planned_activities.append(activity_specs)
        self.walk_manager(60 + 50)
        self.assertTrue((self.activity_list.activity_values[:, [blocked_for_ix], 1] == 0).all(),
                        msg=f"{self.activity_list.activity_values[:, [blocked_for_ix], 1]}")

    def test_stop_activities_with_duration(self):
        self.setup_activity_manager()
        activity_specs = ActivityDescriptorSpecs(1, 0, 50, 0, 0, 0)
        self.activity_manager.planned_activities.append(activity_specs)
        self.walk_manager(80)

        self.assertTrue((self.activity_list.activity_values[:, :, 1] == [0, 0, 0, 50, 0, 0]).all(),
                        msg=f"{self.activity_list.activity_values[:, :, 1]}")

        self.assertTrue((self.activity_list.activity_values[:, :, 0] == [50, 0, 29, 30, 1, 0]).all(),
                        msg=f"{self.activity_list.activity_values[:, :, 0]}")

    def test_stop_activities_with_duration_and_different_default_activity(self):
        self.setup_activity_manager()
        self.activity_manager.current_activity[3:6] = 2
        self.activity_manager.current_default_activity[3:6] = 2
        activity_specs = ActivityDescriptorSpecs(1, 0, 50, 0, 0, 0)
        self.activity_manager.planned_activities.append(activity_specs)
        self.walk_manager(80)

        self.assertTrue((self.activity_list.activity_values[:, :, 1] == [0, 0, 0, 50, 0, 0]).all(),
                        msg=f"{self.activity_list.activity_values[:, :, 1]}")

        none_activity_ids = np.ones_like(self.population.index, dtype=bool)
        none_activity_ids[3:6] = False
        self.assertTrue((self.activity_list.activity_values[none_activity_ids, :, 0] == [50, 0, 29, 30, 1, 0]).all(),
                        msg=f"{self.activity_list.activity_values[:, :, 0]}")
        self.assertTrue((self.activity_list.activity_values[~none_activity_ids, :, 2] == [50, 0, 29, 30, 1, 0]).all(),
                        msg=f"{self.activity_list.activity_values[:, :, 2]}")

    def test_pause_and_resume_activity(self):
        self.setup_activity_manager()
        # Star and activity with duration
        activity_specs = ActivityDescriptorSpecs(1, 0, 50, 0, 0, 0)
        self.activity_manager.planned_activities.append(activity_specs)
        self.walk_manager(30)

        # Queue another activity with duration
        activity_specs = ActivityDescriptorSpecs(3, 0, 50, 0, 0, 0)
        self.activity_manager.planned_activities.append(activity_specs)
        self.walk_manager(20)

        # Check that everything matches
        self.assertTrue((self.activity_list.current_activity == 3).all())
        self.assertTrue((self.activity_manager.interrupted_activities.queue[:, :, 0] ==
                         ActivityDescriptorSpecs(1, 0, 20).specifications).all(),
                        msg=f"{self.activity_manager.interrupted_activities.queue[:, :, 0]}")

        self.walk_manager(40)
        # Check that everything resumed correctly
        self.assertTrue((self.activity_list.current_activity == 1).all(),
                        msg=f"Expecting current activity "
                            f"to be 1 not {self.activity_list.current_activity}")
        self.assertTrue((self.activity_manager.interrupted_activities.num_items == 0).all(),
                        msg=f"{self.activity_manager.interrupted_activities.num_items}")

    def test_check_activity_resource_availability(self):
        self.setup_activity_mgr_with_world()

        activity_specs = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[0].id)
        self.activity_manager.planned_activities[3:6].append(activity_specs)
        occupied_resources = self.activity_manager.check_activity_resource_availability(slice(3, 6),
                                                                                        self.activity_manager.planned_activities)
        expected = [False] * 3
        self.assertListEqual(list(occupied_resources), expected)

        # Test with blocking location It should be available for the first user, but blocked for everyone else
        self.walk_manager(60)
        # Block the living room
        activity_specs = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[0].regions[0].id,
                                                 blocks_location=True)

        self.activity_manager.planned_activities[3:6].append(activity_specs)
        self.activity_manager.planned_activities[3:6].append(activity_specs)
        occupied_resources = self.activity_manager.check_activity_resource_availability(
            slice(3, 6),
            self.activity_manager.planned_activities)

        expected = [False, True, True]

        # Test that the location is effectively blocked for everyone after
        self.assertListEqual(list(occupied_resources), expected)

        # Test that only the blocking agent is at location
        expected = self.activity_manager.current_location_id
        expected[3] = self.w.universe.regions[0].regions[0].id
        expected[4:6] = self.w.universe.regions[0].get_entrance_sub_region().id
        self.assertListEqual(list(expected), list(self.activity_manager.current_location_id))

        self.walk_manager(2)
        activity_specs = ActivityDescriptorSpecs(3, 0, 50, location_id=self.w.universe.regions[0].regions[0].id,
                                                 blocks_location=True)
        activity_specs2 = ActivityDescriptorSpecs(2, 0, 50, location_id=self.w.universe.regions[0].regions[0].id)
        self.activity_manager.planned_activities[3:6].append(activity_specs)
        self.activity_manager.planned_activities[6:9].append(activity_specs2)
        occupied_resources = self.activity_manager.check_activity_resource_availability(
            slice(3, 9),
            self.activity_manager.planned_activities)

        expected = [True] * 6
        self.assertListEqual(list(occupied_resources), expected)

    def test_check_activity_resource_availability_blocking_parent(self):
        self.setup_activity_mgr_with_world()

        activity_specs = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[1].regions[0].id,
                                                 blocks_parent_location=True)
        activity_specs1 = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[1].regions[1].id,
                                                  blocks_parent_location=True)
        activity_specs2 = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[0].regions[0].id,
                                                  blocks_parent_location=True)
        activity_specs3 = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[0].id)
        self.activity_manager.planned_activities[0].append(activity_specs)
        self.activity_manager.planned_activities[1].append(activity_specs1)
        self.activity_manager.planned_activities[3:6].append(activity_specs2)
        self.activity_manager.planned_activities[6:9].append(activity_specs3)

        selector = self.activity_manager.planned_activities.num_items > 0
        occupied_resources = self.activity_manager.check_activity_resource_availability(
            selector,
            self.activity_manager.planned_activities)
        expected = [False, True, False] + [True] * 5
        self.assertListEqual(list(occupied_resources), expected)

    def test_motion_to_correct_location(self):
        self.setup_activity_mgr_with_world()

        activity_specs = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[1].regions[0].id)
        self.activity_manager.planned_activities[:4].append(activity_specs)
        self.walk_manager(10)
        self.assertEqual(self.w.universe.regions[1].regions[0].id, self.population.location[0].id)

    def test_blocked_location(self):
        self.setup_activity_mgr_with_world()

        activity_specs = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[1].regions[0].id,
                                                 blocks_location=True)
        activity_specs2 = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[0].regions[0].id,
                                                  blocks_location=True)
        activity_specs3 = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[2].id,
                                                  blocks_location=True)
        self.activity_manager.planned_activities[0].append(activity_specs)
        self.activity_manager.planned_activities[1].append(activity_specs2)
        self.activity_manager.planned_activities[2].append(activity_specs3)
        self.walk_manager(20)

        # Create baseline
        baseline_blocked_ids = [
            self.w.universe.regions[0].regions[0].id,
            self.w.universe.regions[1].regions[0].id,
            self.w.universe.regions[2].id
            ]

        # Prepare test list
        test_blocked_ids = list(self.activity_manager.location_ids[self.activity_manager.location_blocked, 0])
        self.assertListEqual(baseline_blocked_ids, test_blocked_ids)

    def test_activity_ends_and_unblocks_location_and_parent(self):
        self.setup_activity_mgr_with_world()

        activity_specs = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[1].regions[0].id,
                                                 blocks_location=True, blocks_parent_location=True)
        activity_specs2 = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[0].regions[0].id,
                                                  blocks_location=True, blocks_parent_location=True)
        activity_specs3 = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[2].id,
                                                  blocks_location=True)
        self.activity_manager.planned_activities[0].append(activity_specs)
        self.activity_manager.planned_activities[1].append(activity_specs2)
        self.activity_manager.planned_activities[2].append(activity_specs3)
        self.walk_manager(10)

        self.assertEqual(19, self.activity_manager.location_blocked.sum())
        self.walk_manager(60)

        # Blocking has undesired side-effects. # TODO: Implement locking parents correctly
        warn("Blocking side effect not corrected.")
        self.assertEqual(2, self.activity_manager.location_blocked.sum())

    def test_postpone_activity(self):
        self.setup_activity_mgr_with_world()

        activity_specs = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[0].regions[0].id,
                                                 blocks_location=True)

        self.activity_manager.planned_activities[3:6].append(activity_specs)
        self.walk_manager(20)
        self.assertEqual(1, self.activity_manager.location_blocked.sum())

        self.activity_manager.planned_activities[7].append(activity_specs)
        self.assertListEqual(list(self.activity_manager.planned_activities.queue[7, :, 0]),
                             list(activity_specs.specifications.ravel()))
        self.walk_manager(20)
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[3, :, 0]),
                             [-1] * len(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[4, :, 0]),
                             list(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[5, :, 0]),
                             list(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[7, :, 0]),
                             list(activity_specs.specifications.ravel()))

        test_list = [-1] * len(self.population)
        test_list[7] = 1
        test_list[4:6] = [1] * 2
        self.assertListEqual(list(self.activity_manager.postponed_activities.act_idx), test_list)

        self.walk_manager(20)
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[3, :, 0]),
                             [-1] * len(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[4, :, 0]),
                             [-1] * len(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[5, :, 0]),
                             list(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[7, :, 0]),
                             list(activity_specs.specifications.ravel()))

        self.walk_manager(50)
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[3, :, 0]),
                             [-1] * len(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[4, :, 0]),
                             [-1] * len(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[5, :, 0]),
                             [-1] * len(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[7, :, 0]),
                             list(activity_specs.specifications.ravel()))

        self.walk_manager(50)
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[3, :, 0]),
                             [-1] * len(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[4, :, 0]),
                             [-1] * len(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[5, :, 0]),
                             [-1] * len(activity_specs.specifications.ravel()))
        self.assertListEqual(list(self.activity_manager.postponed_activities.queue[7, :, 0]),
                             [-1] * len(activity_specs.specifications.ravel()))

    def setup_activity_mgr_with_world(self):
        self.setup_world()
        self.setup_activity_manager()
        self.activity_manager.world = self.w.universe
        self.activity_manager.register_available_locations()

    def test_consume_triggered(self):
        self.setup_activity_mgr_with_world()

        activity_specs = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[0].regions[1].id)
        self.activity_manager.triggered_activities[0].append(activity_specs)
        self.walk_manager(20)

        expected_activity = [0] * len(self.population)
        expected_activity[0] = 1
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_activity, current_activity)

        self.walk_manager(40)
        expected_activity[0] = 0
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_activity, current_activity)

    def test_consume_priority(self):
        self.setup_activity_mgr_with_world()

        activity_specs1 = ActivityDescriptorSpecs(1, 0, 50, location_id=self.w.universe.regions[0].regions[1].id)
        activity_specs2 = ActivityDescriptorSpecs(2, 0, 50, block_for=250, location_id=self.w.universe.regions[
            0].regions[1].id)
        self.activity_manager.planned_activities[0].append(activity_specs1)
        self.activity_manager.triggered_activities[0].append(activity_specs2)
        self.walk_manager(20)

        expected_activity = [0] * len(self.population)
        expected_activity[0] = 2
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_activity, current_activity)

        self.walk_manager(40)
        expected_activity[0] = 1
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_activity, current_activity)

        self.walk_manager(60)
        expected_activity[0] = 0
        current_activity = list(self.activity_manager.current_activity)
        self.assertListEqual(expected_activity, current_activity)
