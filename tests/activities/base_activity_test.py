from unittest import TestCase

import numpy as np

from i2mb.activities.activity_queue import ActivityQueue
from i2mb.activities.atomic_activities import Sleep, Work
from i2mb.activities.base_activity import ActivityList
from i2mb.activities.base_activity_descriptor import ActivityDescriptorQueue, ActivityDescriptorSpecs
from i2mb.engine.agents import AgentList


class ActivityListTest(TestCase):
    def setUp(self) -> None:
        self.population_size = 10
        self.population = AgentList(self.population_size)

    def init_list_and_test(self):
        activities = np.random.choice([Sleep, Work], size=self.population_size)
        activities = [c(self.population) for c in activities]
        activity_list = ActivityList(self.population)
        activity_list.activities[:] = np.array(activities).reshape(-1, 1)
        test_pattern = np.array(activities).reshape(-1, 1)
        return activity_list, test_pattern

    def init_population_list_and_test(self):
        activity_list, test_pattern = self.init_list_and_test()
        return activity_list, test_pattern, self.population

    def test_creation(self):
        activity_list, test_pattern = self.init_list_and_test()
        self.assertTrue((activity_list.activities == test_pattern).all(),
                        msg=f"Activity list:\n{activity_list}\nTestPattern:\n{test_pattern}")

    def test_start(self):
        activity_list, test_pattern, population = self.init_population_list_and_test()

        activity_list.start(15)
        start_status = np.array(list(map(lambda x: x.started, activity_list.activities.ravel())))
        test_start = [15] * self.population_size
        self.assertTrue((start_status == test_start).all(),
                        msg=f"Start times:\n{start_status}\nTestPattern:\n{test_start}")

    def test_update_time(self):
        activity_list, test_pattern, population = self.init_population_list_and_test()

        for i in range(5):
            activity_list.update_time()

        elapsed_status = np.array(list(map(lambda x: x.elapsed, activity_list.activities.ravel())))
        accumulated_status = np.array(list(map(lambda x: x.accumulated, activity_list.activities.ravel())))
        test_start = [5] * self.population_size
        self.assertTrue((elapsed_status == test_start).all(),
                        msg=f"Elapsed times:\n{elapsed_status}\nTestPattern:\n{test_start}")

        self.assertTrue((accumulated_status == test_start).all(),
                        msg=f"Accumulated times:\n{accumulated_status}\nTestPattern:\n{test_start}")

    def test_stop(self):
        activity_list, test_pattern, population = self.init_population_list_and_test()

        activity_list.start(15)
        for i in range(5):
            activity_list.update_time()

        activity_list.stop()
        start_status = np.array(list(map(lambda x: x.started, activity_list.activities.ravel())))
        elapsed_status = np.array(list(map(lambda x: x.elapsed, activity_list.activities.ravel())))
        accumulated_status = np.array(list(map(lambda x: x.accumulated, activity_list.activities.ravel())))

        test_start = [None] * self.population_size
        self.assertTrue((start_status == test_start).all(),
                        msg=f"Start times:\n{start_status}\nTestPattern:\n{test_start}")

        test_start = [0] * self.population_size
        self.assertTrue((elapsed_status == test_start).all(),
                        msg=f"Elapsed times:\n{elapsed_status}\nTestPattern:\n{test_start}")

        test_start = [5] * self.population_size
        self.assertTrue((accumulated_status == test_start).all(),
                        msg=f"Accumulated times:\n{accumulated_status}\nTestPattern:\n{test_start}")

    def test_access_view(self):
        activity_list, test_pattern, population = self.init_population_list_and_test()
        activity_list_view = activity_list[3:6]
        test_pattern = test_pattern[3:6]
        self.assertTrue((activity_list_view.activities == test_pattern).all(),
                        msg=f"Activity View:\n{str(activity_list_view)}\nTest Pattern:\n{test_pattern}")

    def test_stop_view(self):
        """This test starts updates and stops a view of the activity list. Therefore it is huge"""
        activity_list, test_pattern, population = self.init_population_list_and_test()
        activity_list_view = activity_list[3:6]
        activity_list_view.start(15)
        for i in range(5):
            activity_list_view.update_time()

        elapsed_status = np.array(list(map(lambda x: x.elapsed, activity_list_view.activities.ravel())))
        test_start = np.array([0] * self.population_size)
        test_start[3:6] = 5
        self.assertTrue((elapsed_status == test_start[3:6]).all(),
                        msg=f"Elapsed times:\n{elapsed_status}\nTestPattern:\n{test_start}")

        activity_list_view.stop()

        start_status = np.array(list(map(lambda x: x.started, activity_list_view.activities.ravel())))
        elapsed_status = np.array(list(map(lambda x: x.elapsed, activity_list_view.activities.ravel())))
        accumulated_status = np.array(list(map(lambda x: x.accumulated, activity_list_view.activities.ravel())))

        test_start = np.array([0] * self.population_size, dtype=object)
        test_start[3:6] = None
        self.assertTrue((start_status == test_start[3:6]).all(),
                        msg=f"Start times:\n{start_status}\nTestPattern:\n{test_start}")

        test_start = np.array([0] * self.population_size)
        test_start[3:6] = 0
        self.assertTrue((elapsed_status == test_start[3:6]).all(),
                        msg=f"Elapsed times:\n{elapsed_status}\nTestPattern:\n{test_start}")

        test_start = np.array([5] * self.population_size)
        test_start[3:6] = 5
        self.assertTrue((accumulated_status == test_start[3:6]).all(),
                        msg=f"Accumulated times:\n{accumulated_status}\nTestPattern:\n{test_start}")


class ActivityQueueTest(TestCase):
    def setUp(self) -> None:
        self.population_size = 10
        self.queue_size = 6

    def init_queue_and_test_pattern(self):
        activity_queue = ActivityQueue(self.population_size, depth=self.queue_size)
        for i in range(1, self.queue_size + 2):
            activity_queue.push(i)

        test_array = np.array([list(range(self.queue_size+1, 1, -1))] * self.population_size, dtype=object)
        self.assertTrue((activity_queue.num_items == activity_queue.len).all(),
                        msg=f"num_items: {activity_queue.num_items[0]}, Queue length: {activity_queue.len}")
        return activity_queue, test_array

    def test_access(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        self.assertTrue((activity_queue.queue == test_array).all())

    def test_view_access(self):
        # test access via View
        activity_queue, test_array = self.init_queue_and_test_pattern()
        self.assertTrue((activity_queue[0, :].queue == test_array[0, :]).all())
        return test_array

    def test_pushing_into_view(self):
        # test pushing on the view
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue[3:6].push(12)
        activity_queue[3:6].push(11)
        new_row = test_array[0, :].ravel().copy()
        new_row[2:] = new_row[:-2]
        new_row[:2] = [11, 12]
        test_array[3:6, :] = new_row
        self.assertTrue((activity_queue.queue == test_array).all())

    def test_jagged_push_end(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        for i in range(activity_queue.len):
            for j in range(activity_queue.len - i):
                activity_queue[[i]].pop()

            test_array[i, :i] = test_array[i, activity_queue.len - i:]
            test_array[i, i:] = None
            test_array[i, i] = 15

        activity_queue.append(15)
        self.assertTrue((activity_queue.queue == test_array).all())

    def test_push_end(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue.pop()
        activity_queue.append(0)
        test_array[:, :-1] = test_array[:, 1:]
        test_array[:, -1] = 0
        self.assertTrue((activity_queue.queue == test_array).all())

    def test_push_end_when_full(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue.append(0)
        self.assertTrue((activity_queue.queue == test_array).all())
        self.assertTrue((activity_queue.num_items == activity_queue.len).all())

    def test_push_end_into_view(self):
        # test pushing on the view
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue[3:6].push(12)
        activity_queue[3:6].push(11)
        activity_queue[3:6].pop()
        activity_queue[3:6].append(0)

        new_row = test_array[0, :].ravel().copy()
        new_row[1:] = new_row[:-1]
        new_row[:1] = [12]
        new_row[-1] = 0
        test_array[3:6, :] = new_row
        self.assertTrue((activity_queue.queue == test_array).all(),
                        msg=f"{activity_queue.queue}\nTest: \n{test_array}")

    def test_pop(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        # test popping full queue
        r = activity_queue.pop()
        self.assertTrue((r == test_array[:, 0]).all())
        self.assertTrue((activity_queue.queue[:, :-1] == test_array[:, 1:]).all())
        self.assertTrue((activity_queue.queue[:, -1] == None).all())

        r = activity_queue.pop()
        self.assertTrue((activity_queue.num_items == activity_queue.len - 2).all())

    def test_pop_from_view(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        test_r = test_array[3:6, 0].copy()
        test_array[3:6, :-1] = test_array[3:6, 1:]
        test_array[3:6, -1] = None

        # test popping form view
        r = activity_queue[3:6].pop()
        self.assertTrue((r == test_r).all())
        self.assertTrue((activity_queue.queue == test_array).all(),
                        msg=f"Activity queue:\n{activity_queue}\nTestPattern:\n{test_array}")

        self.assertTrue((activity_queue.num_items[3:6] == activity_queue.len - 1).all())

    def init_population_queue_and_test_pattern(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        population = AgentList(10)
        population.add_property("activity_queue", activity_queue)
        return activity_queue, test_array, population

    def test_population_access(self):
        activity_queue, test_array, population = self.init_population_queue_and_test_pattern()
        self.assertTrue((population.activity_queue.queue == test_array).all())

    def test_population_view_access(self):
        activity_queue, test_array, population = self.init_population_queue_and_test_pattern()

        self.assertTrue((population[3:6].activity_queue.queue == test_array[3:6]).all())
        self.assertTrue((population.activity_queue[3:6].queue == test_array[3:6]).all())

    def test_push_to_population_view(self):
        activity_queue, test_array, population = self.init_population_queue_and_test_pattern()

        population[3:6].activity_queue.push(12)
        population[3:6].activity_queue.push(11)
        new_row = test_array[0, :].ravel().copy()
        new_row[2:] = new_row[:-2]
        new_row[:2] = [11, 12]
        test_array[3:6, :] = new_row
        self.assertTrue((activity_queue.queue == test_array).all())
        self.assertTrue((population.activity_queue.queue == test_array).all())
        self.assertTrue((population[3:6].activity_queue.queue == test_array[3:6]).all())

    def test_pop_from_population_view(self):
        activity_queue, test_array, population = self.init_population_queue_and_test_pattern()
        test_r = test_array[3:6, 0].copy()
        test_array[3:6, :-1] = test_array[3:6, 1:]
        test_array[3:6, -1] = None

        # test popping form view
        r = population[3:6].activity_queue.pop()
        self.assertTrue((r == test_r).all())
        self.assertTrue((population.activity_queue.queue == test_array).all(),
                        msg=f"Activity queue:\n{activity_queue}\nTestPattern:\n{test_array}")

    def test_reset(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue.reset()
        self.assertTrue((activity_queue.queue == None).all())

    def test_reset_view(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue[3:6].reset()
        test_array[3:6] = None

        self.assertTrue((activity_queue.queue == test_array).all(),
                        msg=f"Activity queue:\n{activity_queue}\nTestPattern:\n{test_array}")


class ActivityDescriptorQueueTest(TestCase):
    def setUp(self) -> None:
        self.population_size = 10
        self.queue_size = 6

    def init_queue_and_test_pattern(self):
        activity_queue = ActivityDescriptorQueue(self.population_size, depth=self.queue_size)
        for i in range(1, self.queue_size + 2):
            activity_queue.push([i] * ActivityDescriptorQueue.num_descriptor_properties)

        test_array = np.array([[list(range(self.queue_size+1, 1, -1))] *
                               ActivityDescriptorQueue.num_descriptor_properties] * self.population_size
                              , dtype=int)
        self.assertTrue((activity_queue.num_items == activity_queue.len).all(),
                        msg=f"num_items: {activity_queue.num_items[0]}, Queue length: {activity_queue.len}")
        return activity_queue, test_array

    def test_access(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        self.assertTrue((activity_queue.queue == test_array).all())

    def test_view_access(self):
        # test access via View
        activity_queue, test_array = self.init_queue_and_test_pattern()
        self.assertTrue((activity_queue[3:7].queue == test_array[3:7, :, :]).all())
        return test_array

    def test_pushing_into_view(self):
        # test pushing on the view
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue[3:6].push(12)
        activity_queue[3:6].push(11)
        new_row = test_array[0, :].copy()
        new_row[:, 2:] = new_row[:, :-2]
        new_row[:, :2] = np.tile([[11], [12]], ActivityDescriptorQueue.num_descriptor_properties, ).T
        test_array[3:6, :] = new_row
        self.assertTrue((activity_queue.queue == test_array).all())

    def test_jagged_push_end(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        for i in range(activity_queue.len):
            for j in range(activity_queue.len - i):
                activity_queue[[i]].pop()

            test_array[i, :, :i] = test_array[i, :, activity_queue.len - i:]
            test_array[i, :, i:] = ActivityDescriptorQueue.empty_slot
            test_array[i, :, i] = 15

        activity_queue.append(15)
        self.assertTrue((activity_queue.queue == test_array).all())

    def test_push_end(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue.pop()
        activity_queue.append(0)
        test_array[:, :, :-1] = test_array[:, :, 1:]
        test_array[:, :, -1] = 0
        self.assertTrue((activity_queue.queue == test_array).all())

    def test_push_end_when_full(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue.append(0)
        self.assertTrue((activity_queue.queue == test_array).all())
        self.assertTrue((activity_queue.num_items == activity_queue.len).all())

    def test_push_end_into_view(self):
        # test pushing on the view
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue[3:6].push(12)
        activity_queue[3:6].push(11)
        activity_queue[3:6].pop()
        activity_queue[3:6].append(0)

        new_row = test_array[0, :].copy()
        new_row[:, 1:] = new_row[:, :-1]
        new_row[:, :1] = np.tile([[12]], ActivityDescriptorQueue.num_descriptor_properties, ).T
        new_row[:, -1] = 0
        test_array[3:6, :] = new_row
        self.assertTrue((activity_queue.queue == test_array).all(),
                        msg=f"{activity_queue.queue}\nTest: \n{test_array}")

        self.assertTrue((activity_queue.num_items == self.queue_size).all(),
                        msg=f"{activity_queue.num_items}\nTest: \n{self.queue_size}")

    def test_append_view(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue.reset()
        activity_queue[3:6].push(10)
        activity_queue[3:6].push(11)
        activity_queue[3:6].append(13)

        test_array[:] = ActivityDescriptorQueue.empty_slot
        test_array[slice(3, 6), :, 0] = 11
        test_array[slice(3, 6), :, 1] = 10
        test_array[slice(3, 6), :, 2] = 13

        self.assertTrue((activity_queue.queue == test_array).all(),
                        msg=f"{activity_queue.queue}\nTest: \n{test_array}")

        self.assertTrue((activity_queue.num_items[3:6] == 3).all(),
                        msg=f"{activity_queue.num_items}\nExpecting: \n{3}")

    def test_append_view_ids(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue.reset()
        activity_queue[[3, 6]].push(10)
        activity_queue[[3, 6]].push(11)
        activity_queue[[3, 6]].append(13)

        test_array[:] = ActivityDescriptorQueue.empty_slot
        test_array[[3, 6], :, 0] = 11
        test_array[[3, 6], :, 1] = 10
        test_array[[3, 6], :, 2] = 13

        self.assertTrue((activity_queue.queue == test_array).all(),
                        msg=f"{activity_queue.queue}\nTest: \n{test_array}")

        self.assertTrue((activity_queue.num_items[[3, 6]] == 3).all(),
                        msg=f"\n{activity_queue.queue}\n{activity_queue.num_items}\nExpecting: \n{3}")

    def test_append(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue.reset()
        activity_queue.push(10)
        activity_queue.push(11)
        activity_queue.append(13)

        test_array[:] = ActivityDescriptorQueue.empty_slot
        test_array[:, :, 0] = 11
        test_array[:, :, 1] = 10
        test_array[:, :, 2] = 13

        self.assertTrue((activity_queue.queue == test_array).all(),
                        msg=f"{activity_queue.queue}\nTest: \n{test_array}")

        self.assertTrue((activity_queue.num_items == 3).all(),
                        msg=f"{activity_queue.num_items}\nTest: \n{self.queue_size}")

    def test_pop(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        # test popping full queue
        r = activity_queue.pop()
        self.assertTrue((r == test_array[:, :, 0]).all())
        self.assertTrue((activity_queue.queue[:, :, :-1] == test_array[:, :, 1:]).all())
        self.assertTrue((activity_queue.queue[:, :, -1] == ActivityDescriptorQueue.empty_slot).all())

        r = activity_queue.pop()
        self.assertTrue((activity_queue.num_items == activity_queue.len - 2).all())

    def test_pop_from_view(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        test_r = test_array[3:6, :, 0].copy()
        test_array[3:6, :, :-1] = test_array[3:6, :, 1:]
        test_array[3:6, :, -1] = ActivityDescriptorQueue.empty_slot

        # test popping form view
        r = activity_queue[3:6].pop()
        self.assertTrue((r == test_r).all())
        self.assertTrue((activity_queue.queue == test_array).all(),
                        msg=f"Activity queue:\n{activity_queue}\nTestPattern:\n{test_array}")

        self.assertTrue((activity_queue.num_items[3:6] == activity_queue.len - 1).all())

    def init_population_queue_and_test_pattern(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        population = AgentList(10)
        population.add_property("activity_queue", activity_queue)
        return activity_queue, test_array, population

    def test_population_access(self):
        activity_queue, test_array, population = self.init_population_queue_and_test_pattern()
        self.assertTrue((population.activity_queue.queue == test_array).all())

    def test_population_view_access(self):
        activity_queue, test_array, population = self.init_population_queue_and_test_pattern()

        self.assertTrue((population[3:6].activity_queue.queue == test_array[3:6]).all())
        self.assertTrue((population.activity_queue[3:6].queue == test_array[3:6]).all())

    def test_push_to_population_view(self):
        activity_queue, test_array, population = self.init_population_queue_and_test_pattern()

        population[3:6].activity_queue.push(12)
        population[3:6].activity_queue.push(11)
        new_row = test_array[0, :].copy()
        new_row[:, 2:] = new_row[:, :-2]
        new_row[:, :2] = np.tile([[11], [12]], ActivityDescriptorQueue.num_descriptor_properties, ).T
        test_array[3:6, :] = new_row
        self.assertTrue((activity_queue.queue == test_array).all())
        self.assertTrue((population.activity_queue.queue == test_array).all())
        self.assertTrue((population[3:6].activity_queue.queue == test_array[3:6]).all())

    def test_pop_from_population_view(self):
        activity_queue, test_array, population = self.init_population_queue_and_test_pattern()
        test_r = test_array[3:6, :, 0].copy()
        test_array[3:6, :, :-1] = test_array[3:6, :, 1:]
        test_array[3:6, :, -1] = ActivityDescriptorQueue.empty_slot

        # test popping form view
        r = population[3:6].activity_queue.pop()
        self.assertTrue((r == test_r).all())
        self.assertTrue((population.activity_queue.queue == test_array).all(),
                        msg=f"Activity queue:\n{activity_queue}\nTestPattern:\n{test_array}")

    def test_reset(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue.reset()
        self.assertTrue((activity_queue.queue == ActivityDescriptorQueue.empty_slot).all())

    def test_reset_view(self):
        activity_queue, test_array = self.init_queue_and_test_pattern()
        activity_queue[3:6].reset()
        test_array[3:6] = 0

        self.assertTrue((activity_queue.queue == test_array).all(),
                        msg=f"Activity queue:\n{activity_queue}\nTestPattern:\n{test_array}")

    def test_push_using_activity_specs(self):
        activity_queue, test_array, population = self.init_population_queue_and_test_pattern()
        activity_specs1 = ActivityDescriptorSpecs(act_idx=[12])
        activity_specs2 = ActivityDescriptorSpecs(act_idx=[11])
        population[3:6].activity_queue.push(activity_specs1)
        population[3:6].activity_queue.push(activity_specs2)
        new_row = test_array[0, :].copy()
        new_row[:, 2:] = new_row[:, :-2]
        new_row[:, :2] = np.vstack([activity_specs2.specifications, activity_specs1.specifications]).T
        test_array[3:6, :] = new_row
        self.assertTrue((activity_queue.queue == test_array).all())
        self.assertTrue((population.activity_queue.queue == test_array).all())
        self.assertTrue((population[3:6].activity_queue.queue == test_array[3:6]).all())

    def test_pop_using_activity_specs(self):
        activity_queue, test_array, population = self.init_population_queue_and_test_pattern()
        activity_specs1 = ActivityDescriptorSpecs(act_idx=[12])
        activity_specs2 = ActivityDescriptorSpecs(act_idx=[11])
        population[3:6].activity_queue.push(activity_specs1)
        population[3:6].activity_queue.push(activity_specs2)
        new_row = test_array[0, :].copy()
        new_row[:, 2:] = new_row[:, :-2]
        new_row[:, :2] = np.vstack([activity_specs2.specifications, activity_specs1.specifications]).T
        test_array[3:6, :] = new_row
        self.assertTrue((activity_queue.queue == test_array).all())
        self.assertTrue((population.activity_queue.queue == test_array).all())
        self.assertTrue((population[3:6].activity_queue.queue == test_array[3:6]).all())

        act_specs = activity_queue[3:6].pop()
        self.assertEqual(act_specs.shape, (3, ActivityDescriptorQueue.num_descriptor_properties))
        self.assertTrue((activity_specs2.specifications == act_specs).all())
