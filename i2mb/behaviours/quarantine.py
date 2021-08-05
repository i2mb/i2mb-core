import numpy as np
from i2mb import Model
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time


class QuarantineBehaviour(Model):
    def __init__(self, population: AgentList, quarantine_start, quarantine_time):

        self.population = population
        self.quarantine_time = quarantine_time
        self.quar_start = quarantine_start

        self.bedrooms = {}
        self.beds = {}
        self.sleep_pos = {}
        self.dress_pos = {}

        n = len(self.population)
        self.in_quarantine = np.zeros((n, 1), dtype=bool)
        self.has_schedule = np.zeros(n, dtype=bool)
        self.quarantine_start = np.full((n, 1), np.inf)
        self.accumulated_quarantine = np.zeros((n, 1))

        population.add_property("in_quarantine", self.in_quarantine)
        population.add_property("accumulated_quarantine", self.accumulated_quarantine)

    def step(self, t):
        unhealthy = (self.population.state >= 4).ravel()
        if unhealthy.any:
            new_schedule = unhealthy & (~self.has_schedule).ravel()
            if new_schedule.any():
                day_offset = global_time.time_scalar * global_time.day(t)
                start = np.full((len(self.population), 1), day_offset)
                start[new_schedule] += self.quar_start

                self.quarantine_start[new_schedule] = start[new_schedule]
                self.has_schedule[new_schedule] = True

        # start quarantine
        start_quarantine = self.quarantine_start < t
        start_quarantine = start_quarantine & (~self.in_quarantine)
        if start_quarantine.any():
            self.in_quarantine[start_quarantine] = True
            # change sleeping pos for other particle not infected
            same_bedroom = [self.population.bedroom == x for x in self.population.bedroom[start_quarantine.ravel()]]
            idx = list(self.population.index[same_bedroom])
            idx_quarantine = self.population.index[self.population.in_quarantine.ravel()]
            idx.remove(idx_quarantine)
            if len(idx) > 0:
                self.bedrooms.update(dict.fromkeys(self.population.bedroom[idx], idx))
                apartment = self.population.home[idx]
                self.population.bedroom[idx] = [a.living_room for a in apartment]
                self.sleep_pos.update(dict.fromkeys(idx, self.population.sleep_pos[idx]))
                self.population.sleep_pos[idx] = [a.living_room.sitting_pos[:len(idx)] for a in apartment]

                self.dress_pos.update(dict.fromkeys(idx, self.population.dress_pos[idx]))
                self.population.dress_pos[idx] = [a.living_room.target_pos[:len(idx)] for a in apartment]

                self.population.motion_mask[start_quarantine & ~self.population.in_bed] = True

        # update quarantine_time
        if self.in_quarantine.any():
            if hasattr(self.population, "eat"):
                self.population.eat[self.in_quarantine] = False
            if hasattr(self.population, "working"):
                self.population.working[self.in_quarantine] = False
            if hasattr(self.population, "bath"):
                self.population.bath[self.in_quarantine] = False
            self.accumulated_quarantine[self.in_quarantine] += 1

        # let people go outside again
        enough_quarantine = (self.accumulated_quarantine > self.quarantine_time).ravel()
        if enough_quarantine.any():
            self.in_quarantine[enough_quarantine] = False
            self.accumulated_quarantine[enough_quarantine] = 0
            self.quarantine_start[enough_quarantine] = np.inf
            self.has_schedule[enough_quarantine] = False

            for bedroom in self.population.bedroom[enough_quarantine]:
                idx_bedroom = self.bedrooms[bedroom]
                del self.bedrooms[bedroom]
                self.population.bedroom[idx_bedroom] = bedroom

                idx_bedroom = idx_bedroom[0]
                sleep_pos = self.sleep_pos[idx_bedroom]
                del self.sleep_pos[idx_bedroom]
                self.population.sleep_pos[idx_bedroom] = sleep_pos

                dress_pos = self.dress_pos[idx_bedroom]
                del self.dress_pos[idx_bedroom]
                self.population.sleep_pos[idx_bedroom] = dress_pos
        return
