import numpy as np

from i2mb.activities.base_activity import ActivityPrimitive


class Sleep(ActivityPrimitive):
    def __init__(self, population):
        super().__init__(population)
        n = len(self.population)
        self.last_wakeup_time = np.full((n, 1), -np.inf)
        self.sleep = np.zeros((n, 1), dtype=bool)
        self.in_bed = np.zeros((n, 1), dtype=bool)
        self.has_plan = np.zeros(n, dtype=bool)
        self.beds = np.zeros((n, 2), dtype=int)

        self.population.add_property("sleep", self.sleep)
        self.population.add_property("in_bed", self.in_bed)

        self.rank = 1

    def finalize_start(self, ids):
        if len(ids) == 0:
            return

        locations = set(self.population.location[ids].ravel())
        for loc in locations:
            if not hasattr(loc, "put_to_bed"):
                continue

            ids_mask = self.population.location[ids] == loc
            idx = self.population.index[ids]
            loc.put_to_bed(idx[ids_mask])

    def start_activity(self, t, start_activity_selector):
        return self.put_people_to_bed(t, start_activity_selector)

    def put_people_to_bed(self, t, send_to_bed):
        # send_to_bed = (self.sleep & ~self.in_bed).ravel()
        if not send_to_bed.any():
            return np.array([]), np.array([])

        at_home_mask = self.population.at_home
        if not at_home_mask.any():
            return np.array([]), np.array([])

        send_to_bed &= at_home_mask
        if send_to_bed.any():
            self.in_bed[send_to_bed] = True
            self.sleep[send_to_bed] = True
            if hasattr(self.population, "position"):
                self.population.position[send_to_bed] = self.beds[send_to_bed]
                self.population.motion_mask[send_to_bed] = False

        return

    def stop_activity(self, t, ids, descriptors_ids):
        self.has_plan[ids] = False
        awaken = self.wake_people_up(t, ids)
        self.get_people_out_of_bed(awaken)
        return awaken

    def wake_people_up(self, t, ids):
        # Wake people up
        # enough_sleep = ids
        # left_home = ~self.population.at_home.ravel() & self.in_bed.ravel()
        # wake_up = enough_sleep | left_home[ids]
        if len(ids) > 0:
            self.sleep[ids] = False
            self.last_wakeup_time[ids] = t

        return ids

    def get_people_out_of_bed(self, ids):
        get_out_of_bed = ids
        if len(get_out_of_bed) == 0:
            return np.array([], dtype=int)

        # self.values[:, self.in_progress_ix][get_out_of_bed] = False
        locations = set(self.population.location[get_out_of_bed].ravel())
        ids = self.population.index[get_out_of_bed]
        for loc in locations:
            if not hasattr(loc, "get_out_of_bed"):
                continue

            loc.get_out_of_bed(ids)

        return self.population.index[get_out_of_bed]


class Work(ActivityPrimitive):
    def __init__(self, population):
        super().__init__(population)

        # Let the location determine the motion status
        self.stationary = None


class Eat(ActivityPrimitive):
    pass


class Rest(ActivityPrimitive):
    pass


class Toilet(ActivityPrimitive):
    pass


class Sink(ActivityPrimitive):
    pass


class Shower(ActivityPrimitive):
    pass


class Cook(ActivityPrimitive):
    pass
