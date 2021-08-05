import numpy as np
from i2mb import Model
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time
from functools import partial


class EatBehaviour(Model):
    def __init__(self, population: AgentList, eating_duration, eating_start, prepare_duration):
        self.eating_start = eating_start
        self.eating_duration = eating_duration
        self.preparing_duration = prepare_duration

        self.population = population

        n = len(self.population)

        self.day = np.full((n, 1), -np.inf)
        self.morning = np.ones((n,), dtype=bool)
        self.ate_dinner = np.zeros((n,), dtype=bool)

        self.eat = np.zeros((n, 1), dtype=bool)
        self.is_eating = np.zeros((n, 1), dtype=bool)
        self.has_schedule = np.zeros(n, dtype=bool)
        self.eat_start = np.full((n, 1), np.inf)

        self.current_eating_duration = np.full((n, 1), np.inf)
        self.accumulated_eating = np.zeros((n, 1))
        self.last_eating_time = np.full((n, 1), -np.inf)

        self.current_preparing_duration = np.full((n, 1), np.inf)
        self.is_preparing = np.zeros((n, 1), dtype=bool)
        self.accumulated_prepare = np.zeros((n, 1))
        self.ready_prepare = np.zeros((n, 1), dtype=bool)

        population.add_property("eat", self.eat)
        population.add_property("is_eating", self.is_eating)
        population.add_property("is_preparing", self.is_preparing)
        population.add_property("ready_prepare", self.ready_prepare)

    def step(self, t):
        new_schedule = ~self.has_schedule

        if new_schedule.any():
            self.morning = (self.day < global_time.days(t)).ravel()
            morning = self.morning & new_schedule
            if morning.any():
                eat_start = self.eating_start((morning.sum(), 1))
                eat_duration = self.eating_duration((morning.sum(), 1)) / 2
                prepare_duration = self.preparing_duration((morning.sum(), 1)) / 2

                day_offset = global_time.time_scalar * global_time.days(t)
                if 0 < t < global_time.make_time(day=1):
                    day_offset = global_time.time_scalar
                start = day_offset + eat_start

                start[start < self.last_eating_time[morning]] += global_time.time_scalar
                self.eat_start[morning] = start.astype(int)
                self.current_eating_duration[morning] = eat_duration + prepare_duration

                # time in kitchen preparing meal
                self.current_preparing_duration[morning] = prepare_duration
                self.has_schedule[morning] = True
                self.day[morning] = global_time.days(t)
                self.ate_dinner[morning] = False

            evening = ~self.morning & new_schedule & ~self.ate_dinner

            if evening.any():
                eat_start = self.eating_start((evening.sum(), 1))
                eat_duration = self.eating_duration((evening.sum(), 1))
                prepare_duration = self.preparing_duration((evening.sum(), 1))

                day_offset = global_time.time_scalar * (global_time.days(t))
                offset = (partial(np.random.normal, global_time.make_time(hour=11), global_time.make_time(minutes=30)))(
                    (evening.sum(), 1))
                start = day_offset + eat_start + offset

                self.eat_start[evening] = start.astype(int)
                self.current_eating_duration[evening] = eat_duration + prepare_duration

                # time in kitchen preparing meal
                self.current_preparing_duration[evening] = prepare_duration
                self.has_schedule[evening] = True
                self.morning[evening] = False
                self.ate_dinner[evening] = True

        start_eat = (self.eat_start <= t)
        starting = start_eat.ravel() & (~self.population.busy).ravel() & ~self.eat.ravel()
        if starting.any():
            self.population.eat[starting] = True
            self.population.busy[starting] = True

        # Update eating duration
        eat = self.eat & (self.is_eating | self.is_preparing)
        if eat.any():
            self.accumulated_eating[eat] += 1
        prepare = self.eat & self.is_preparing
        if prepare.any():
            self.accumulated_prepare[prepare] += 1

        # Make people stop preparing
        enough_prepared = (self.accumulated_prepare > self.current_preparing_duration).ravel()
        has_prepared = self.is_preparing.ravel()
        stop_preparing = enough_prepared & has_prepared
        if stop_preparing.any():
            self.ready_prepare[stop_preparing] = True
            self.accumulated_prepare[stop_preparing] = 0
            self.current_preparing_duration[stop_preparing] = np.inf

        # Make people stop eating
        enough_eaten = (self.accumulated_eating > self.current_eating_duration).ravel()
        had_a_meal = self.is_eating.ravel()
        stop_eating = (enough_eaten & had_a_meal)
        if stop_eating.any():
            self.eat[stop_eating] = False
            self.last_eating_time[stop_eating] = t
            self.accumulated_eating[stop_eating] = 0
            self.eat_start[stop_eating] = np.inf
            self.current_eating_duration[stop_eating] = np.inf
            self.ready_prepare[stop_eating] = False
            self.has_schedule[stop_eating] = False
        return
