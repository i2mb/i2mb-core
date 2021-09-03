import numpy as np

from i2mb.activities.activity_queue import ActivityQueue


class RoutineCollection:
    pass


class Routine:
    def __init__(self, population_size, activity_indices, activity_list, padding_index=0):
        self.activity_list = activity_list
        self.activity_indices = activity_indices
        self.queue = ActivityQueue(population_size, len(activity_indices), padding_index)
        self.reset_routine_queue()

    def reset_routine_queue(self, ids=None, skip_activities=None):
        if skip_activities is None:
            skip_activities = []

        try:
            skip_activities = np.array(skip_activities, dtype=int)
        except TypeError as e:
            raise TypeError("skip_activities must be a list of integer activity indices. Original error:\n" + f"{e}")

        if ids is None:
            ids = slice(None)

        reset_depth = len(self.activity_indices) - len(skip_activities)
        q_length = len(self.queue.queue[ids, :])
        activity_mask = np.setdiff1d(self.activity_indices, skip_activities)
        new_activities = np.tile(activity_mask, q_length).reshape(q_length, reset_depth)
        for r in range(q_length):
            np.random.shuffle(new_activities[r, :])

        self.queue.queue[ids, :reset_depth] = new_activities

    def get_next_activity(self, ids):
        activities = self.queue[ids].pop()
        return activities
