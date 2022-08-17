from typing import TYPE_CHECKING

import numpy as np

from i2mb.activities import ActivityDescriptorProperties
from i2mb.activities.base_activity import ActivityNone
from i2mb.activities.base_activity_descriptor import ActivityDescriptorSpecs

if TYPE_CHECKING:
    from i2mb.activities.activity_manager import ActivityManager
    from i2mb.worlds import World


class DefaultActivityController:
    z_order = -1  # order of creation dictates importance

    def __init__(self, population):
        super().__init__()

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
                activity_manager.activity_controllers[region.default_activity.activity_class.id] = self

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

    @staticmethod
    def has_new_activity(inactive_ids):
        return inactive_ids

    def get_new_activity(self, inactive_ids):
        default_descriptor = self.current_default_activity_descriptor[inactive_ids, :]
        return default_descriptor, np.ones_like(inactive_ids, dtype=bool)

    def stage_default_activities_on_entry(self, idx, region, arriving_from):
        if not hasattr(region, "default_activity"):
            return

        self.current_default_activity_descriptor[idx, :] = region.default_activity.create_specs().specifications

        # Default activities, should be interruptable, and repeatable. Therefore, they should not block nor have a
        # defined duration.
        self.current_default_activity_descriptor[idx, ActivityDescriptorProperties.duration] = 0
        self.current_default_activity_descriptor[idx, ActivityDescriptorProperties.block_for] = 0

    def cancel_default_activity_on_exit(self, idx, region):
        self.current_default_activity_descriptor[idx, :] = ActivityDescriptorSpecs(ActivityNone.id,
                                                                                   size=len(idx)).specifications
