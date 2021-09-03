from typing import Union

import numpy as np

from i2mb import Model
from i2mb.activities.base_activity import ActivityList


class ActivityManager(Model):
    def __init__(self, population, world, activities: Union[ActivityList, None] = None):
        if activities is None:
            activities = ActivityList(population)

        self.activities = activities
        self.world = world
        self.population = population

        self.current_activity = self.activities.current_activity
        self.current_activity_rank = np.zeros_like(self.current_activity)

        self.activity_ranking = {}

    def post_init(self):
        self.generate_activity_ranking()

    def step(self, t):
        self.update_current_activity()
        self.trigger_activity_stop(t)
        self.trigger_activity_start(t)

    def trigger_activity_start(self, t):
        ranks = len(self.activity_ranking)
        for rank in range(ranks):
            for ix, act in self.activity_ranking[rank]:
                ids, locations = act.start_activity(t, self.current_activity_rank)

                if len(ids) == 0:
                    continue

                self.relocate_agents(ids, locations)
                act.finalize_start(ids)
                self.current_activity[ids] = ix
                self.current_activity_rank[ids] = act.rank

    def trigger_activity_stop(self, t):
        for act in self.activities.activities:
            ids = act.stop_activity(t)
            if len(ids) == 0:
                continue

            self.current_activity[ids] = 0
            self.current_activity_rank[ids] = 0

    def update_current_activity(self):
        # Update sleep duration
        elapsed_ix = self.activities.elapsed_ix
        accumulated_ix = self.activities.accumulated_ix

        property_value = self.activities.get_current_activity_property(elapsed_ix)
        self.activities.set_current_activity_property(elapsed_ix, property_value + 1)

        property_value = self.activities.get_current_activity_property(accumulated_ix)
        self.activities.set_current_activity_property(accumulated_ix, property_value + 1)

    def relocate_agents(self, ids, locations):
        unique_locations = set(locations)
        for loc in unique_locations:
            self.world.move_agents(ids[locations == loc], loc)

    def generate_activity_ranking(self):
        for ix, act in enumerate(self.activities.activities):
            self.activity_ranking.setdefault(act.rank, list()).append((ix, act))


