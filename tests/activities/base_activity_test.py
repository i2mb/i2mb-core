from unittest import TestCase

import numpy as np

import i2mb.activities.atomic_activities as aa

from i2mb.activities import ActivityProperties
from i2mb.activities.base_activity import ActivityList
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time


class TestActivityList(TestCase):
    def setUp(self) -> None:
        self.population_size = 10
        self.activity_classes = [aa.Work, aa.Grooming, aa.Shower, aa.KitchenWork, aa.Rest]
        self.population = AgentList(self.population_size)
        global_time.set_sim_time(0)

    def setup_activity_list(self):
        self.activity_list = ActivityList(self.population)
        for activity in self.activity_classes:
            self.activity_list.add(activity(self.population))

    def populate_activity_list(self, activity_list):
        # Modify the list values
        for i in range(len(self.activity_classes) + 1):
            for j in range(len(ActivityProperties)):
                activity_list.activity_values[:, j, i] = j + i * 100

    def test_activity_register(self):
        activity_list = ActivityList(self.population)
        activity_classes = [aa.Work, aa.Grooming, aa.Shower, aa.KitchenWork, aa.Rest]
        activities = []
        for activity_cls in self.activity_classes:
            activity = activity_cls(self.population)
            activity_list.add(activity)
            activities.append(activity)

        self.populate_activity_list(activity_list)

        # Check that changes are reflected via the individual activity accesses
        for activity in activities:
            for prop in ActivityProperties:
                activity_props = activity.__getattribute__(f"get_{prop.name}")()
                self.assertTrue((activity_props ==
                                activity_list.activity_values[:, prop, activity.id]).all(),
                                msg=f"Activity values {activity_props}")

    def test_activity_property_getters(self):
        self.setup_activity_list()
        self.populate_activity_list(self.activity_list)
        act_id = 2
        act_ids = [1, 2, 3]
        ids = np.array([1, 2, 3, 4, 8])
        unique_activities = np.array([5, 0, 4, 2, 1])

        for prop in ActivityProperties:
            activity_props = self.activity_list.__getattribute__(f"get_{prop.name}")(ids, slice(None))
            self.assertTrue((activity_props == [prop + 100 * a_i.id for a_i in self.activity_list.activities]).all(),
                            msg=f"{activity_props}")

            activity_props = self.activity_list.__getattribute__(f"get_{prop.name}")(slice(None), act_id)
            self.assertTrue((activity_props == prop + 100 * act_id).all(),
                            msg=f"{activity_props}")

            activity_props = self.activity_list.__getattribute__(f"get_{prop.name}")(ids, act_id)
            self.assertTrue((activity_props == prop + 100 * act_id).all(),
                            msg=f"{activity_props}")

            activity_props = self.activity_list.__getattribute__(f"get_{prop.name}")(slice(None), act_ids)
            self.assertTrue((activity_props == [prop + 100 * a_i for a_i in act_ids]).all(),
                            msg=f"{activity_props}")

            activity_props = self.activity_list.__getattribute__(f"get_{prop.name}")(ids.reshape(-1, 1), act_ids)
            self.assertTrue((activity_props == [prop + 100 * a_i for a_i in act_ids]).all(),
                            msg=f"{activity_props}")

            activity_props = self.activity_list.__getattribute__(f"get_{prop.name}")(ids, unique_activities)
            self.assertTrue((activity_props == [prop + 100 * a_i for a_i in unique_activities]).all(),
                            msg=f"{activity_props}")

    def test_activity_property_setters(self):
        self.setup_activity_list()
        act_id = 2
        act_ids = [1, 2, 3]
        ids = np.array([1, 2, 3, 4, 8])
        unique_activities = np.array([5, 0, 4, 2, 1])

        for prop in ActivityProperties:
            self.activity_list.__getattribute__(f"set_{prop.name}")(ids, slice(None), 5)
            activity_props = self.activity_list.activity_values[ids, prop, :]
            self.assertTrue((activity_props == 5).all(),
                            msg=f"{activity_props}")

            self.activity_list.activity_values[:] = 0
            self.activity_list.__getattribute__(f"set_{prop.name}")(slice(None), act_id, 5)
            activity_props = self.activity_list.activity_values[:, prop, act_id]
            self.assertTrue((activity_props == 5).all(),
                            msg=f"{activity_props}")

            self.activity_list.activity_values[:] = 0
            self.activity_list.__getattribute__(f"set_{prop.name}")(slice(None), act_ids, 5)
            activity_props = self.activity_list.activity_values[:, prop, act_id]
            self.assertTrue((activity_props == 5).all(),
                            msg=f"{activity_props}")

            self.activity_list.activity_values[:] = 0
            self.activity_list.__getattribute__(f"set_{prop.name}")(ids.reshape(-1, 1), act_ids, 5)
            activity_props = self.activity_list.activity_values[:, prop, act_ids]
            self.assertFalse((activity_props == 5).all(), msg=f"{activity_props}")
            activity_props = self.activity_list.activity_values[ids, prop, :]
            self.assertFalse((activity_props == 5).all(), msg=f"{activity_props}")
            activity_props = self.activity_list.activity_values[ids.reshape(-1, 1), prop, act_ids]
            self.assertTrue((activity_props == 5).all(), msg=f"{activity_props}")

            self.activity_list.activity_values[:] = 0
            self.activity_list.__getattribute__(f"set_{prop.name}")(ids, unique_activities, 5)
            activity_props = self.activity_list.activity_values[ids.reshape(-1, 1), prop, act_ids]
            expected = np.zeros((len(ids), len(act_ids)))
            expected[3, 1] = 5
            expected[4, 0] = 5
            self.assertTrue((activity_props == expected).all(),
                            msg=f"{activity_props}\n{expected}")




