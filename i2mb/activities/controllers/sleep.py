import numpy as np

from i2mb import Model
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.atomic_activities import Sleep
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time
from i2mb.worlds import World


class SleepBehaviourController(Model):
    def __init__(self, population: AgentList, activity_manager: ActivityManager, world: World,  sleep_duration,
                 sleep_midpoint):

        self.population = population
        self.activity_manager = activity_manager
        self.world = world
        self.sleep_midpoint = sleep_midpoint
        self.sleep_duration = sleep_duration

        self.sleep_activity = Sleep(self.population)
        self.activity_manager.activities.register(self.sleep_activity)
        self.last_wakeup_time = self.sleep_activity.last_wakeup_time
        self.plan_dispatched = np.zeros(len(self.population), dtype=bool)

        self.sleep_profiles = None
        self.create_sleep_profiles()

    def step(self, t):
        self.clear_sleep_activity(t)
        self.update_sleep_schedule(t)
        self.make_people_sleepy(t)
        return

    def update_sleep_schedule(self, t):
        has_schedule = self.sleep_activity.has_plan
        new_schedule = ~has_schedule
        if new_schedule.any():
            sleep_midpoints = self.sleep_midpoint((new_schedule.sum()))
            sleep_durations = self.sleep_duration((new_schedule.sum()))

            day_offset = global_time.time_scalar * (global_time.days(t) + 1)
            start = day_offset + sleep_midpoints - sleep_durations // 2
            if t == 0:
                start -= global_time.time_scalar
                sleep_durations += start

            self.sleep_profiles.specifications[new_schedule, 1] = start
            self.sleep_profiles.specifications[new_schedule, 2] = sleep_durations

            # # Make un interruptable
            self.sleep_profiles.specifications[new_schedule, 4] = 15
            self.sleep_activity.has_plan[new_schedule] = True
            self.plan_dispatched[new_schedule] = False

    def make_people_sleepy(self, t):
        make_sleepy = (self.sleep_profiles.specifications[:, 1] <= t)
        make_sleepy &= ~self.plan_dispatched
        if make_sleepy.any():
            activity_specs = self.sleep_profiles.specifications[make_sleepy, :]
            self.activity_manager.triggered_activities[make_sleepy].append(activity_specs)
            self.plan_dispatched[make_sleepy] = True

    def create_sleep_profiles(self):
        if self.sleep_profiles is None:
            act_idx = self.activity_manager.activities.activity_types.index(type(self.sleep_activity))
            self.sleep_profiles = ActivityDescriptorSpecs(act_idx=act_idx,
                                                          block_for=global_time.make_time(hour=3),
                                                          size=len(self.population))

            self.sleep_profiles.specifications[:, 5] = -1
            for r in self.world.list_all_regions():
                if hasattr(r, "agent_bedroom"):
                    ids = r.bed_assignment.ravel()
                    self.sleep_profiles.specifications[ids, 5] = [br.id for br in r.agent_bedroom]
                    self.sleep_activity.beds[ids, :] = [bed.get_activity_position() for bed in r.beds]

    def clear_sleep_activity(self, t):
        sleep_walkers = self.activity_manager.current_activity == self.sleep_activity.id
        sleep_walkers &= ~self.sleep_activity.sleep.ravel()
        if sleep_walkers.any():
            self.activity_manager.stop_activities(sleep_walkers, t)






