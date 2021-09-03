import numpy as np

from i2mb.motion.base_motion import Motion
from i2mb.pathogen import UserStates
from i2mb.utils import global_time

_1700 = global_time.make_time(hour=17)
_0200 = global_time.make_time(hour=2)


class NightOut(Motion):
    def __init__(self, world, population, group_location, venues, duration, arrival, min_capacity=0.5,
                 opening_hours=_1700, closing_hours=_0200):
        super().__init__(world, population)

        # Eventually this goes to the restaurants and bars
        self.time_sheet = {}
        self.destinations = np.zeros(len(population), dtype=object)
        self.leave_time = np.zeros(len(population), dtype=int)
        self.return_time = np.zeros(len(population), dtype=int)
        self.going_out = np.zeros(len(population), dtype=bool)
        self.groups = []

        self.opening_hours = opening_hours
        self.closing_hours = closing_hours
        if closing_hours < opening_hours:
            self.closing_hours += global_time.make_time(hour=24)

        self.min_capacity = min_capacity
        self.max_capacity = sum([v.sits for v in venues])
        if self.max_capacity > len(self.population):
            self.max_capacity = int(len(self.population) * .7)

        self.arrival = arrival
        self.duration = duration
        self.venues = venues
        self.venues_open = False
        self.group_location = group_location

    def step(self, t):
        time = global_time.hour(t), global_time.minute(t)
        if time == (6, 0):
            self.plan_night_out(t)
            return

        if (~self.going_out).all():
            return

        if t == self.opening_hours:
            self.venues_open = True

        if not self.venues_open:
            return

        for g in self.time_sheet.get(t, []):
            venue, leave_ix = list(g.items())[0]
            if not venue.can_sit_party(leave_ix):
                if t + global_time.make_time(hour=1) < self.closing_hours:
                    self.time_sheet.setdefault(t + global_time.make_time(hour=1), []).append(g)

                continue

            self.world.move_agents(leave_ix, venue)
            leave_ix = leave_ix[~self.population.remain[leave_ix]]
            venue.sit_particles(leave_ix)

        return_ix = (self.return_time <= t) & (self.population.location.reshape(-1, 1) == self.venues).any(axis=1)

        if t == self.closing_hours:
            self.opening_hours += global_time.make_time(hour=24)
            self.closing_hours += global_time.make_time(hour=24)
            self.venues_open = False
            return_ix = (self.population.location.reshape(-1, 1) == self.venues).any(axis=1)

        if not return_ix.any():
            return

        destinations = set(self.population.home[return_ix])
        for r in destinations:
            self.world.move_agents(return_ix & (self.population.home == r), r)

    def plan_night_out(self, t):
        self.going_out[:] = False
        self.destinations[:] = None
        n = len(self.population)
        move_mask = np.ones(n, dtype=bool)
        if hasattr(self.population, "isolated"):
            move_mask &= ~self.population.isolated.ravel()

        if hasattr(self.population, "state"):
            deceased = self.population.state == UserStates.deceased
            move_mask &= ~deceased.ravel()

        a_population = sum(move_mask)
        max_capacity = self.max_capacity
        if max_capacity > a_population:
            max_capacity = a_population

        if a_population < self.min_capacity:
            return

        going_out = np.random.randint(self.min_capacity, max_capacity)
        going_out_ix = np.random.choice(self.population.index[move_mask], going_out, replace=False)

        groups = np.random.choice(self.group_location, going_out)
        venue = np.random.choice(self.venues, going_out)
        self.destinations[going_out_ix] = venue
        self.going_out[going_out_ix] = True
        going_with = np.zeros_like(self.destinations)
        going_with[going_out_ix] = groups

        # Create groups
        self.groups = []
        for g in self.group_location:
            g_locations = getattr(self.population, g)
            for g_loc in set(g_locations[going_out_ix]):
                for v in set(venue):
                    ix = self.population.index[self.going_out & (g_locations == g_loc) &
                                               (going_with == g) & (self.destinations == v)]
                    if len(ix) == 0:
                        continue

                    self.groups.extend([{v: ix[s:s + 10]} for s in range(0, len(ix), 10)])

        # Assign time
        self.time_sheet = {}
        opening_time = self.opening_hours
        for g in self.groups:
            ix = list(g.values())[0]
            arrival = global_time.day(t) * global_time.time_scalar + self.arrival()
            if arrival < opening_time:
                arrival = opening_time + 4

            self.leave_time[ix] = arrival
            self.return_time[ix] = self.duration(len(ix)) + arrival
            self.time_sheet.setdefault(arrival, []).append(g)
