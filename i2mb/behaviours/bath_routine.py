import numpy as np
from i2mb import Model
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time


class BathBehaviour(Model):
    def __init__(self, population: AgentList, bath_duration, short_duration, bath_start):
        self.bathing_start = bath_start
        self.bath_duration = bath_duration
        self.short_duration = short_duration
        self.population = population

        n = len(self.population)
        self.bath = np.zeros((n, 1), dtype=bool)
        self.in_bathroom = np.zeros((n, 1), dtype=bool)
        self.has_schedule = np.zeros(n, dtype=bool)
        self.bath_start = np.full((n, 1), np.inf)
        self.current_bath_duration = np.full((n, 1), np.inf)
        self.accumulated_bath = np.zeros((n, 1))
        self.last_bath_time = np.full((n, 1), -np.inf)

        self.day = np.full((n, 1), -np.inf)

        population.add_property("bath", self.bath)
        population.add_property("in_bathroom", self.in_bathroom)

    def step(self, t):
        time = global_time.hour(t)

        new_schedule = ~self.has_schedule
        # if time == 0:
        #   new_schedule = np.ones(len(self.population), dtype=bool)
        if new_schedule.any():
            bath_start = self.bathing_start((new_schedule.sum(), 1))
            new_day = self.day < global_time.days(t)
            start = bath_start
            start[start < self.last_bath_time[new_schedule]] += t
            self.bath_start[new_schedule] = start.astype(int)
            shower = new_day.ravel() & new_schedule
            not_shower = ~new_day.ravel() & new_schedule
            if shower.any():
                durations = self.bath_duration((shower.sum(), 1))
                self.current_bath_duration[shower] = durations
                self.day[shower] = global_time.days(t)
            if not_shower.any():
                durations = self.short_duration((not_shower.sum(), 1))
                self.current_bath_duration[not_shower] = durations
            self.has_schedule[new_schedule] = True

        make_dirty = (self.bath_start <= t).ravel()
        take_a_bath = make_dirty & (~self.population.busy).ravel() & (~self.bath).ravel()
        if take_a_bath.any():
            self.bath[take_a_bath] = True
            self.population.busy[take_a_bath] = True

        # Update bath duration
        cleaning = self.bath & self.in_bathroom
        if cleaning.any():
            self.accumulated_bath[cleaning] += 1

        # Let people get dirty
        enough_cleaning = (self.accumulated_bath > self.current_bath_duration).ravel()
        at_home = self.population.at_home.ravel() & self.in_bathroom.ravel()
        enough_cleaning = enough_cleaning & at_home
        enough_cleaning = enough_cleaning
        if enough_cleaning.any():
            self.bath[enough_cleaning] = False
            self.last_bath_time[enough_cleaning] = t
            self.accumulated_bath[enough_cleaning] = 0
            self.bath_start[enough_cleaning] = np.inf
            self.current_bath_duration[enough_cleaning] = np.inf
            self.has_schedule[enough_cleaning] = False

        return
