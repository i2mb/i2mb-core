from unittest import TestCase

from i2mb.activities.atomic_activities import *
from i2mb.activities.base_activity import ActivityList
from i2mb.activities.routines import Routine
from i2mb.engine.agents import AgentList


class RoutineTest(TestCase):
    def setUp(self) -> None:
        self.population_size = 10
        self.population = AgentList(self.population_size)
        self.activity_list = ActivityList(self.population)

        activities = np.random.choice([Sleep, Work, Cook, Eat, Toilet, Shower, Sink], size=self.population_size)
        activities = [c(self.population) for c in activities]
        self.activities = activities
        for act in activities:
            self.activity_list.register(act)

        activities.insert(0, self.activity_list.activities[0])

    def test_creation(self):
        routine = Routine(self.activity_list)
        self.assertEqual(routine.queue.size, self.population_size)
        self.assertEqual(routine.queue.len, len(self.activity_list.activities))
        routine = Routine(self.activity_list, [0, 1, 2])

        return




