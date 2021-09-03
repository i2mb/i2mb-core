import numpy as np

from i2mb.activities.base_activity import ActivityPrimitive
from i2mb.utils import int_inf


class Sleep(ActivityPrimitive):
    def __init__(self, population):
        super().__init__(population)
        n = len(self.population)
        self.last_wakeup_time = np.full((n, 1), -np.inf)
        self.sleep = np.zeros((n, 1), dtype=bool)
        self.in_bed = np.zeros((n, 1), dtype=bool)

        self.population.add_property("sleep", self.in_bed)
        self.population.add_property("in_bed", self.in_bed)

        self.rank = 1

    def finalize_start(self, ids):
        if len(ids) == 0:
            return

        locations = set(self.population.location[ids].ravel())
        for loc in locations:
            ids_mask = self.population.location[ids] == loc
            loc.put_to_bed(ids[ids_mask])

    def start_activity(self, t, ranks):
        return self.put_people_to_bed(t)

    def put_people_to_bed(self, t):
        send_to_bed = (self.sleep & ~self.in_bed).ravel()
        if not send_to_bed.any():
            return np.array([]), np.array([])

        at_home_mask = self.population.at_home
        if not at_home_mask.any():
            return np.array([]), np.array([])

        send_to_bed &= at_home_mask
        locations = set(self.population.location[send_to_bed].ravel())
        self.values[:, self.in_progress_ix][send_to_bed] = True

        go_to_bedrooms = np.zeros_like(send_to_bed, dtype=object)
        go_to_bedrooms[:] = None
        for loc in locations:
            home = self.population.home == loc.parent
            mask = send_to_bed & home
            ids = self.population.index[mask]
            bed_mask = (ids == loc.parent.bed_assignment).any(axis=1)
            go_to_bedrooms[mask] = loc.parent.agent_bedroom[bed_mask]

        return self.population.index[send_to_bed], go_to_bedrooms[send_to_bed]

    def stop_activity(self, t):
        awaken = self.wake_people_up(t)
        self.get_people_out_of_bed(t)
        return awaken

    def wake_people_up(self, t):
        # Wake people up
        enough_sleep = (self.get_elapsed() > self.get_duration()).ravel()
        left_home = ~self.population.at_home.ravel() & self.in_bed.ravel()
        wake_up = enough_sleep | left_home
        if wake_up.any():
            self.sleep[wake_up] = False
            self.last_wakeup_time[wake_up] = t
            self.get_elapsed()[wake_up] = 0
            self.get_start()[wake_up] = int_inf
            self.get_duration()[wake_up] = 0
            self.get_in_progress()[wake_up] = False

        return wake_up

    def get_people_out_of_bed(self, t):
        get_out_of_bed = (~self.sleep & self.in_bed).ravel()
        if not get_out_of_bed.any():
            return np.array([], dtype=int)

        self.values[:, self.in_progress_ix][get_out_of_bed] = False
        locations = set(self.population.location[get_out_of_bed].ravel())
        ids = self.population.index[get_out_of_bed]
        for loc in locations:
            if not hasattr(loc, "get_out_of_bed"):
                continue


            loc.get_out_of_bed(ids)

        return self.population.index[get_out_of_bed]


class Work(ActivityPrimitive):
    pass


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
