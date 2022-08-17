import numpy as np

from i2mb import Model
from i2mb.activities import ActivityDescriptorProperties
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.atomic_activities import Sleep
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time, time
from i2mb.worlds import World


class SleepBehaviourController(Model):
    z_order = 0  # make the first on execution queues

    def __init__(self, population: AgentList, activity_manager: ActivityManager, sleep_duration,
                 sleep_midpoint, minimum_up_time: int = 3):

        super().__init__()

        self.population = population
        self.activity_manager = activity_manager
        self.sleep_midpoint = sleep_midpoint
        self.sleep_duration = sleep_duration
        self.minimum_up_time = minimum_up_time

        self.sleep_activity = Sleep(self.population)
        self.last_wakeup_time = self.sleep_activity.last_wakeup_time
        self.plan_dispatched = np.zeros(len(self.population), dtype=bool)
        self.make_sleepy = np.zeros(len(self.population), dtype=bool)
        self.bedrooms = np.zeros(len(self.population), dtype=object)
        self.has_plan = np.zeros(len(self.population), dtype=bool)
        self.sleep_profiles = None

    def post_init(self, base_file_name=None):
        homes = set(self.population.home)
        for home in homes:
            self.bedrooms[home.inhabitants.index] = [r.index for r in home.agent_bedroom.ravel()]

    def step(self, t):
        self.clear_sleep_activity(t)
        self.update_sleep_schedule(t)

    def registration_callback(self, activity_manager: 'ActivityManager', world: 'World'):
        self.sleep_activity = activity_manager.register_activity(self.sleep_activity)
        activity_manager.activity_controllers[self.sleep_activity.id] = self
        self.create_sleep_profiles(activity_manager, world)
        # self.register_location_trigger_actions(activity_manager, world)
        self.register_on_action_stop()
        self.register_on_action_start()

    def update_sleep_schedule(self, t):
        new_schedule = ~self.has_plan
        if new_schedule.any():
            sleep_midpoints = self.sleep_midpoint((new_schedule.sum()))
            sleep_durations = self.sleep_duration((new_schedule.sum()))

            day_offset = global_time.time_scalar * (global_time.days(t) + 1)
            start = day_offset + sleep_midpoints - sleep_durations // 2
            if t == 0:
                start -= global_time.time_scalar
                sleep_durations += start

            self.sleep_profiles.specifications[new_schedule, ActivityDescriptorProperties.start] = start
            self.sleep_profiles.specifications[new_schedule, ActivityDescriptorProperties.duration] = sleep_durations

            # # Make un interruptable by adding a blocking_for 15 value
            self.sleep_profiles.specifications[new_schedule, ActivityDescriptorProperties.block_for] = self.minimum_up_time
            self.has_plan[new_schedule] = True
            self.plan_dispatched[new_schedule] = False

    def has_new_activity(self, inactive_ids):
        self.make_sleepy = (self.sleep_profiles.specifications[:, ActivityDescriptorProperties.start] <= time())
        self.make_sleepy &= ~self.plan_dispatched
        self.make_sleepy &= self.population.at_home.ravel()
        return self.make_sleepy

    def get_new_activity(self, ids):
        if self.make_sleepy.any():
            activity_specs = self.sleep_profiles.specifications[self.make_sleepy, :]
            activity_specs[:, ActivityDescriptorProperties.location_ix] = self.get_bedrooms()
            return activity_specs, self.make_sleepy[ids]

        return np.zeros_like(ids, dtype=object), np.zeros_like(ids, dtype=bool)

    def create_sleep_profiles(self, activity_manager, world):
        if self.sleep_profiles is None:
            act_idx = activity_manager.activity_list.activity_types.index(type(self.sleep_activity))
            self.sleep_profiles = ActivityDescriptorSpecs(act_idx=act_idx,
                                                          block_for=global_time.make_time(hour=self.minimum_up_time),
                                                          interruptable=False,
                                                          size=len(self.population))

            self.sleep_profiles.specifications[:, 5] = -1
            for r in world.list_all_regions():
                if hasattr(r, "agent_bedroom"):
                    ids = r.bed_assignment.ravel()
                    self.sleep_profiles.specifications[ids, 5] = [br.id for br in r.agent_bedroom]
                    self.sleep_activity.beds[ids, :] = [bed.get_activity_position() for bed in r.beds]

    def clear_sleep_activity(self, t):
        sleep_walkers = self.activity_manager.current_activity == self.sleep_activity.id
        sleep_walkers &= ~self.sleep_activity.sleep.ravel()
        if sleep_walkers.any():
            self.activity_manager.stop_activities(t, self.population.index[sleep_walkers])

    def reset_sleep_on_stop(self, act_id, t, stop_selector):
        self.plan_dispatched[stop_selector] = False
        self.has_plan[stop_selector] = False

    def marked_dispatched_plans(self, act_id, t, stop_selector):
        self.plan_dispatched[stop_selector] = True

    def register_on_action_stop(self):
        self.sleep_activity.register_stop_callbacks(self.reset_sleep_on_stop)

    def register_on_action_start(self):
        self.sleep_activity.register_start_callbacks(self.marked_dispatched_plans)

    def get_bedrooms(self):
        return self.bedrooms[self.make_sleepy]







