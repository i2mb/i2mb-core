import numpy as np

from i2mb import Model
from i2mb.activities.atomic_activities import Sleep
from i2mb.activities.base_activity import ActivityList
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time


class SleepBehaviour(Model):
    def __init__(self, population: AgentList, activity_list: ActivityList, sleep_duration, sleep_midpoint):
        self.sleep_midpoint = sleep_midpoint
        self.sleep_duration = sleep_duration
        self.population = population

        n = len(self.population)

        self.has_schedule = np.zeros(n, dtype=bool)
        self.sleep_activity = Sleep(self.population)
        activity_list.register(self.sleep_activity)

        self.sleep_start = self.sleep_activity.get_start()
        self.current_sleep_duration = self.sleep_activity.get_duration()
        self.last_wakeup_time = self.sleep_activity.last_wakeup_time
        self.sleep = self.sleep_activity.sleep

    def post_init(self):
        self.sleep_start = self.sleep_activity.get_start()
        self.current_sleep_duration = self.sleep_activity.get_duration()

    def step(self, t):
        self.update_sleep_schedule(t)
        self.make_people_sleepy(t)
        return

    def update_sleep_schedule(self, t):
        has_schedule = self.current_sleep_duration > 0
        new_schedule = ~has_schedule
        if new_schedule.any():
            sleep_midpoints = self.sleep_midpoint((new_schedule.sum(), 1))
            sleep_durations = self.sleep_duration((new_schedule.sum(), 1))

            day_offset = global_time.time_scalar * global_time.days(t)
            start = day_offset + sleep_midpoints - sleep_durations / 2
            start[start < self.last_wakeup_time[new_schedule]] += global_time.time_scalar
            self.sleep_start[new_schedule] = start.astype(int).ravel()
            self.current_sleep_duration[new_schedule] = sleep_durations.ravel()

    def make_people_sleepy(self, t):
        make_sleepy = (self.sleep_start <= t)
        self.sleep[make_sleepy] = True
