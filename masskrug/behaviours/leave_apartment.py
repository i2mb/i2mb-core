import numpy as np
from masskrug import Model
from masskrug.engine.particle import ParticleList
from masskrug.utils import global_time


class LeaveBehaviour(Model):
    def __init__(self, population: ParticleList, outside_duration, outside_start):
        self.outside_start = outside_start
        self.outside_duration = outside_duration
        self.population = population

        n = len(self.population)

        self.outside = np.zeros((n, 1), dtype=bool)
        self.is_outside = np.zeros((n, 1), dtype=bool)
        self.has_schedule = np.zeros(n, dtype=bool)
        self.out_start = np.full((n, 1), np.inf)
        self.current_outside_duration = np.full((n, 1), np.inf)
        self.accumulated_outside = np.zeros((n, 1))
        self.last_outside_time = np.full((n, 1), -np.inf)

        population.add_property("working", self.outside)
        population.add_property("is_outside", self.is_outside)

    def step(self, t):
        new_schedule = ~self.has_schedule
        if new_schedule.any():
            outside_start = self.outside_start((new_schedule.sum(), 1))
            outside_duration = self.outside_duration((new_schedule.sum(), 1))

            day_offset = global_time.time_scalar * global_time.days(t)
            if 0 < t < global_time.make_time(day=1):
                day_offset = global_time.time_scalar
            start = day_offset + outside_start

            start[start < self.last_outside_time[new_schedule]] += global_time.time_scalar
            self.out_start[new_schedule] = start.astype(int)
            self.current_outside_duration[new_schedule] = outside_duration
            self.has_schedule[new_schedule] = True

        go_outside = (self.out_start <= t).ravel()
        ready = (~self.population.busy).ravel() & ~self.outside.ravel()
        if (ready & go_outside).any():
            self.outside[ready & go_outside] = True
            self.population.busy[ready & go_outside] = True

        # Update outside duration
        out = self.outside & self.is_outside
        if out.any():
            self.accumulated_outside[out] += 1

        # Make people come home
        enough_work = (self.accumulated_outside > self.current_outside_duration).ravel()
        left_home = self.is_outside.ravel()
        come_home = enough_work & left_home
        if come_home.any():
            self.outside[come_home] = False
            self.last_outside_time[come_home] = t
            self.accumulated_outside[come_home] = 0
            self.out_start[come_home] = np.inf
            self.current_outside_duration[come_home] = np.inf
            self.has_schedule[come_home] = False

        return
