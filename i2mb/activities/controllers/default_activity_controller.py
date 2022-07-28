from i2mb import Model
from i2mb.activities import ActivityDescriptorProperties
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.base_activity import ActivityNone
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs


class DefaultActivityController(Model):
    def __init__(self, population, activity_manager: ActivityManager):
        super().__init__()

        self.activity_manager = activity_manager
        self.population = population

        # Default activity can be changed during runtime
        self.current_default_activity_descriptor = ActivityDescriptorSpecs(ActivityNone.id,
                                                                           size=len(self.population)).specifications
        self.current_default_activity = self.current_default_activity_descriptor[:, ActivityDescriptorProperties.act_idx]

    def post_init(self, base_file_name=None):
        self.register_enter_actions()
        self.register_on_exit_action()

    def register_enter_actions(self):
        relocator = self.activity_manager.relocator
        if relocator is None:
            return

        relocator.register_on_region_enter_action([
            self.stage_default_activities_on_entry
        ])

    def register_on_exit_action(self):
        relocator = self.activity_manager.relocator
        if relocator is None:
            return

        relocator.register_on_region_exit_action([
            self.cancel_default_activity_on_exit
        ])

    def step(self, t):
        # Handle finished activities
        self.stage_default_activities(t)

    def stage_activities(self, has_activities_to_stage, activity_descriptors_to_stage, t):
        # Start times
        activity_descriptors_to_stage[:, 1] = t

        ids = self.population.index[has_activities_to_stage]
        no_blocked_activities = self.activity_manager.stage_activity(activity_descriptors_to_stage, ids)
        idx = ids[no_blocked_activities]

    def stage_default_activities_on_entry(self, idx, region, arriving_from):
        if not hasattr(region, "default_activity"):
            return

        self.current_default_activity_descriptor[idx, :] = region.default_activity.create_specs().specifications

        # Default activities, should be interruptable, and repeatable. Therefore, they should not block nor have a
        # defined duration.
        self.current_default_activity_descriptor[idx, ActivityDescriptorProperties.duration] = 0
        self.current_default_activity_descriptor[idx, ActivityDescriptorProperties.block_for] = 0

        no_planned_activity = self.activity_manager.current_descriptors[idx, ActivityDescriptorProperties.act_idx] == -1
        if no_planned_activity.any():
            default_descriptor = self.current_default_activity_descriptor[idx, :][no_planned_activity]
            self.activity_manager.stage_activity(default_descriptor, idx[no_planned_activity])

    def cancel_default_activity_on_exit(self, idx, region):
        self.current_default_activity_descriptor[idx, :] = ActivityDescriptorSpecs(ActivityNone.id,
                                                                                   size=len(idx)).specifications

    def stage_default_activities(self, t):
        inactive = self.activity_manager.current_activity == -1
        inactive &= self.activity_manager.current_descriptors[:, ActivityDescriptorProperties.act_idx] == -1
        if inactive.any():
            ids = self.population.index[inactive]
            default_descriptor = self.current_default_activity_descriptor[ids, :]
            self.activity_manager.stage_activity(default_descriptor, ids)
