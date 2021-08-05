import numpy as np

from masskrug import Model
from masskrug.engine.agents import AgentList
from masskrug.utils import global_time


class SleepBehaviour(Model):
    def __init__(self, population: AgentList, sleep_duration, sleep_midpoint):
        self.sleep_midpoint = sleep_midpoint
        self.sleep_duration = sleep_duration
        self.population = population

        n = len(self.population)
        self.sleep = np.zeros((n, 1), dtype=bool)
        self.in_bed = np.zeros((n, 1), dtype=bool)
        self.has_schedule = np.zeros(n, dtype=bool)
        self.sleep_start = np.full((n, 1), np.inf)
        self.current_sleep_duration = np.full((n, 1), np.inf)
        self.accumulated_sleep = np.zeros((n, 1))
        self.last_wakeup_time = np.full((n, 1), -np.inf)

        population.add_property("sleep", self.sleep)
        population.add_property("in_bed", self.in_bed)

    def step(self, t):
        new_schedule = ~self.has_schedule
        if new_schedule.any():
            sleep_midpoints = self.sleep_midpoint((new_schedule.sum(), 1))
            sleep_durations = self.sleep_duration((new_schedule.sum(), 1))

            day_offset = global_time.time_scalar * global_time.days(t)
            start = day_offset + sleep_midpoints - sleep_durations / 2
            if 0 < t < global_time.make_time(day=1):
                day_offset = global_time.time_scalar

            start[start < self.last_wakeup_time[new_schedule]] += global_time.time_scalar
            self.sleep_start[new_schedule] = start.astype(int)
            self.current_sleep_duration[new_schedule] = sleep_durations
            self.has_schedule[new_schedule] = True

        make_sleepy = (self.sleep_start <= t)
        self.sleep[make_sleepy] = True

        # Update sleep duration
        sleeping = self.sleep & self.in_bed
        if sleeping.any():
            self.accumulated_sleep[sleeping] += 1

        # Wake people up
        enough_sleep = (self.accumulated_sleep > self.current_sleep_duration).ravel()
        left_home = (self.population.location != self.population.home) & self.in_bed.ravel()
        wake_up = enough_sleep | left_home
        if wake_up.any():
            self.sleep[wake_up] = False
            self.in_bed[left_home] = False
            self.last_wakeup_time[wake_up] = t
            self.accumulated_sleep[wake_up] = 0
            self.sleep_start[wake_up] = np.inf
            self.current_sleep_duration[wake_up] = np.inf
            self.has_schedule[wake_up] = False

        return
