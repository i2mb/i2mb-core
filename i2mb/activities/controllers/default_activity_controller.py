from typing import TYPE_CHECKING

from i2mb.activities import ActivityDescriptorProperties
from i2mb.activities.base_activity import ActivityNone
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs

if TYPE_CHECKING:
    from i2mb.activities.activity_manager import ActivityManager
    from i2mb.worlds import World


class DefaultActivityController:
    def __init__(self, population, activity_manager: 'ActivityManager'):
        super().__init__()

        self.activity_manager = activity_manager
        self.population = population

        # Default activity can be changed during runtime
        self.current_default_activity_descriptor = ActivityDescriptorSpecs(ActivityNone.id,
                                                                           size=len(self.population)).specifications
        self.current_default_activity = self.current_default_activity_descriptor[:, ActivityDescriptorProperties.act_idx]

    def registration_callback(self, activity_manager: 'ActivityManager', world: 'World'):
        self.register_enter_actions(activity_manager)
        self.register_on_exit_action(activity_manager)
        for region in world.list_all_regions():
            if hasattr(region, "default_activity"):
                key = (region.index, region.default_activity.activity_class.id)
                activity_manager.add_location_activity_controller(*key, self)

    def register_enter_actions(self, activity_manager):
        relocator = activity_manager.relocator
        if relocator is None:
            return

        relocator.register_on_region_enter_action([
            self.stage_default_activities_on_entry
        ])

    def register_on_exit_action(self, activity_manager):
        relocator = activity_manager.relocator
        if relocator is None:
            return

        relocator.register_on_region_exit_action([
            self.cancel_default_activity_on_exit
        ])

    def step_on_handler(self, region):
        # Handle finished activities
        return self.stage_default_activities(region)

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

    def stage_default_activities(self, region):
        inactive = self.activity_manager.current_activity == -1
        inactive &= self.activity_manager.current_descriptors[:, ActivityDescriptorProperties.act_idx] == -1
        inactive &= self.population.location == region
        if inactive.any():
            ids = self.population.index[inactive]
            default_descriptor = self.current_default_activity_descriptor[ids, :]
            return default_descriptor, ids

        return [], []
