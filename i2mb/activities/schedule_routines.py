from itertools import cycle
from typing import Iterable

import numpy as np

from i2mb.motion.base_motion import Motion
from i2mb.pathogen import UserStates
from i2mb.utils import global_time
from i2mb.worlds import World


class Entry:
    def __init__(self, start_time=0, exit_area=None, duration=None,
                 repeats=True, end_time=None, area_error=0.5, event_location=None, auto_return=False,
                 return_to=None):
        trigger = "duration"
        if duration is None:
            if end_time is None:
                if exit_area is None:
                    raise RuntimeError("One of the following must be defined defined: duration, end_time, or exit_area")
                else:
                    trigger = "area"
            else:
                trigger = "end_time"

        self.trigger = np.array([trigger == "duration", trigger == "end_time", trigger == "area"])
        self.repeats = repeats
        self.auto_return = auto_return
        self.return_to = return_to

        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration

        self.event_location = event_location
        if exit_area is None:
            self.exit_area = np.zeros(4) * -1
        else:
            self.exit_area = np.array(exit_area)

        self.area_error = area_error

    @staticmethod
    def from_entry(entry):
        return Entry(entry.start_time, entry.exit_area, entry.duration, entry.repeats, entry.end_time,
                     entry.area_error, entry.event_location, entry.auto_return, entry.return_to)

    def resolve(self, t):
        offset = 0
        if not isinstance(self.start_time, int):
            self.start_time = self.start_time()

        if not isinstance(self.end_time, int) and self.end_time is not None:
            self.end_time = self.end_time()

        if not isinstance(self.duration, int) and self.duration is not None:
            self.duration = self.duration()

        if not isinstance(self.event_location, str) and not isinstance(self.event_location, World) and \
                self.event_location is not None:
            self.event_location = self.event_location()

        if not isinstance(self.return_to, str) and not isinstance(self.return_to, World) and self.return_to is not None:
            self.return_to = self.return_to()

        if self.repeats == "daily" or self.repeats is True:
            offset = global_time.days(t) * global_time.time_scalar
            if self.start_time + offset < t:
                offset += global_time.time_scalar
        elif self.repeats == "weekly":
            offset = global_time.week_start(t)
            if self.start_time + offset < t:
                offset += global_time.ticks_week
        elif self.repeats == "monthly":
            offset = global_time.month_start(t)
            if self.start_time + offset < t:
                offset += global_time.ticks_month

        end_next_period = 0
        if self.end_time is not None and self.start_time > self.end_time:
            end_next_period = 1

        self.start_time += offset
        if self.trigger[0]:  # "duration"
            self.end_time = self.start_time + self.duration
        else:
            self.end_time += offset + end_next_period * offset
            self.duration = self.end_time - self.start_time


class Schedule:
    def __init__(self, collection):
        self.repeats_yearly = []  # Not implemented

        self.repeats_monthly = []
        self.repeats_weekly = []
        self.repeats_daily = []

        self.unique_events = []

        self.__daily_cycle = None
        self.__weekly_cycle = None
        self.__monthly_cycle = None

        self.__latest_unique = None
        self.__latest_daily = None
        self.__latest_weekly = None
        self.__latest_monthly = None
        self.__latest = None

        self.items = []

        self.__deque = []

        self.allocate_events(collection)

    def get_locations(self):
        return np.unique(
            [i.event_location for i in self.items] + [i.return_to for i in self.items if i.return_to is not None])

    def get_next_event(self, t) -> Entry:
        self.update_queues(t)

        most_current_event = None
        for event in [self.__latest_daily, self.__latest_weekly, self.__latest_monthly, self.__latest_unique]:
            event: Entry
            if event is None:
                continue

            if most_current_event is None or most_current_event.start_time > event.start_time:
                most_current_event = event

        return most_current_event

    def allocate_events(self, collection):
        for e in collection:
            if isinstance(e, Entry):
                event = Entry.from_entry(e)
            else:
                event = Entry(**e)

            self.items.append(event)
            if not event.repeats:
                self.unique_events.append(event)

            if event.repeats == "daily" or event.repeats is True:
                self.repeats_daily.append(event)

            if event.repeats == "weekly":
                self.repeats_weekly.append(event)

            if event.repeats == "monthly" or event.repeats is True:
                self.repeats_monthly.append(event)

        self.repeats_daily = sorted(self.repeats_daily, key=lambda x: x.start_time())
        self.repeats_weekly = sorted(self.repeats_weekly, key=lambda x: x.start_time())
        self.repeats_monthly = sorted(self.repeats_monthly, key=lambda x: x.start_time())

        self.__daily_cycle = cycle(self.repeats_daily)
        self.__weekly_cycle = cycle(self.repeats_weekly)
        self.__monthly_cycle = cycle(self.repeats_monthly)

    def update_queues(self, t):
        if self.__latest_unique is None or self.__latest_unique.end_time < t:
            if self.unique_events:
                self.__latest_unique = self.unique_events.pop(0)
                self.__latest_unique.resolve(t)

        if self.__latest_daily is None or self.__latest_daily.end_time < t:
            if self.repeats_daily:
                self.__latest_daily = Entry.from_entry(next(self.__daily_cycle))
                self.__latest_daily.resolve(t)

        if self.__latest_weekly is None or self.__latest_weekly.end_time < t:
            if self.repeats_weekly:
                self.__latest_weekly = Entry.from_entry(next(self.__weekly_cycle))
                self.__latest_weekly.resolve(t)

        if self.__latest_monthly is None or self.__latest_monthly.end_time < t:
            if self.repeats_monthly:
                self.__latest_monthly = Entry.from_entry(next(self.__monthly_cycle))
                self.__latest_monthly.resolve(t)


class ScheduleRoutines(Motion):
    def __init__(self, population, relocator, schedule: [Schedule, Iterable], default="home",
                 must_follow_schedule=1.,
                 ignore_schedule=None):
        super().__init__(population)
        self.relocator = relocator
        n = len(population)

        self.__initialize_must_follow_schedule_selection(must_follow_schedule, n)
        self.__initialize_ignore_schedule_selection(ignore_schedule, must_follow_schedule, n)

        self.default_location = getattr(self.population, default)
        self.schedules = np.empty((n, 1), dtype=object)
        if isinstance(schedule, Schedule):
            # Each particle gets an independent copy.
            self.schedules[:] = np.array([Schedule(schedule.items) for s in range(n)]).reshape((-1, 1))
            schedule = [schedule]
        else:
            self.schedules[:] = schedule

        self.have_schedule = self.schedules != None  # noqa

        self.population.add_property("schedule", self.schedules)
        # self.location_types = np.unique([loc for s in self.schedules.ravel() for loc in s.get_locations()])
        # self.__init_locations__()

        self.event = np.empty((n, 1), dtype=object)

        # triggers => [["duration", "end_time", "area"], ...]
        self.trigger = np.zeros((n, 3), dtype=bool)
        self.start_time = np.zeros((n, 1), dtype=int)
        self.end_time = np.zeros((n, 1), dtype=int)
        self.duration = np.zeros((n, 1), dtype=int)
        self.auto_return = np.zeros((n, 1), dtype=bool)
        self.event_location = np.zeros(n, dtype=object)
        self.return_location = self.default_location.copy()
        self.exit_area = np.zeros((n, 4), dtype=object)

        self.update_events([True] * n, 0)

    def __initialize_ignore_schedule_selection(self, ignore_schedule, must_follow_schedule, n):
        self.ignore_schedule = ignore_schedule
        self.decides_to_follow_schedule_selection = None
        if ignore_schedule is None:
            return

        if 1 - must_follow_schedule < ignore_schedule:
            raise RuntimeError("Config Problem: ignore_schedule must be less than 1 - must_follow_schedule")

        self.decides_to_follow_schedule_selection = np.ones((n, 1), dtype=bool)

    def __initialize_must_follow_schedule_selection(self, must_follow_schedule, n):
        self.must_follow_schedule = must_follow_schedule
        self.can_ignore_schedule_selection = np.ones((n, 1), dtype=bool)
        self.can_ignore_schedule_selection[
            np.random.choice(n, int(must_follow_schedule * n), replace=False)] = False

    # def __init_locations__(self):
    #     n = len(self.population)
    #     for loc in self.location_types:
    #         if hasattr(self.population, loc):
    #             continue
    #
    #         loc_vector = np.empty((n, ), dtype=object)
    #         self.population.add_property(loc, loc_vector)

    def step(self, t):
        self.update_follow_daily_schedule(t)

        # Process triggers
        duration_trigger, end_time_trigger, area_trigger = self.trigger.T
        start_triggers = (self.start_time == t).ravel()
        end_triggers = duration_trigger & ((t - self.start_time) > self.duration).ravel()
        end_triggers |= end_time_trigger & (self.end_time <= t).ravel()
        # noinspection PyUnresolvedReferences
        end_triggers |= (area_trigger &
                         ((self.population.position < self.exit_area[:, 2:]) &
                          (self.population.position > self.exit_area[:, :2])).any(axis=1))

        active_triggers = end_triggers | start_triggers
        if not any(active_triggers):
            return

        n = len(self.population)
        move_mask = np.ones((n, 1), dtype=bool) & self.have_schedule
        if self.ignore_schedule is not None:
            move_mask &= self.decides_to_follow_schedule_selection

        if hasattr(self.population, "isolated"):
            move_mask &= ~self.population.isolated

        deceased = self.population.state == UserStates.deceased
        move_mask &= ~deceased

        go_home = move_mask & self.auto_return & end_triggers.reshape((-1, 1))
        self.event_location[go_home.ravel()] = self.return_location[go_home.ravel()]

        # not_at_location = (self.population.location != self.event_location).reshape((-1, 1))
        # move_mask &= (self.start_time <= t) & not_at_location
        move_mask &= active_triggers.reshape((-1, 1))  # & not_at_location

        regions = set(self.event_location[move_mask.ravel()].ravel())
        for r in regions:
            self.relocator.move_agents(move_mask.ravel() & (self.event_location == r), r)

        self.update_events(end_triggers, t)

    def update_events(self, active_triggers, t):
        for id_, s in zip(self.population.index[active_triggers], self.schedules[active_triggers].ravel()):
            if s is None:
                continue

            event: Entry = s.get_next_event(t)
            self.trigger[id_] = event.trigger
            self.start_time[id_] = event.start_time
            self.end_time[id_] = event.end_time is not None and event.end_time or 0
            self.duration[id_] = event.duration is not None and event.duration or 0
            self.auto_return[id_] = event.auto_return
            if event.return_to is None:
                self.return_location[id_] = self.default_location[id_]
            elif isinstance(event.return_to, str):
                self.return_location[id_] = getattr(self.population, event.return_to)[id_]
            else:
                self.return_location[id_] = event.return_to

            if event.event_location is None:
                self.event_location[id_] = self.default_location[id_]
            elif isinstance(event.event_location, str):
                self.event_location[id_] = getattr(self.population, event.event_location)[id_]
            else:
                self.event_location[id_] = event.event_location

            self.exit_area[id_] = event.exit_area

    def update_follow_daily_schedule(self, t):
        if self.ignore_schedule is None:
            return

        if global_time.hour(t) != 0 or global_time.minute(t) != 5:
            # Decide only at 00:05 if you are
            return

        self.decides_to_follow_schedule_selection[:, :] = True

        ids = self.population.index[self.can_ignore_schedule_selection.ravel()]
        ignore_schedule_size = int(len(self.population) * self.ignore_schedule)
        ignore_schedule = np.random.choice(ids, ignore_schedule_size, replace=False)
        self.decides_to_follow_schedule_selection[ignore_schedule, :] = False


