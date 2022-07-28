import unittest

from tests.world_tester import WorldBuilderTestsNoGui
from tests.core.agent_lists_test import TestAgentList
from tests.motion.random_motion_test import RandomMotion
from tests.activities.base_activity_test import TestActivityList
from tests.activities.activity_manager_test import TestActivityManager
from tests.activities.default_activity_controller_test import TestDefaultActivityController
from tests.activities.sleep_behaviour_test import TestSleepBehaviourNoGui
# from tests.activities.location_activity_controller_test import TestLocationActivityController
# from tests.activities.activity_queue_test import ActivityQueueTest
from tests.activities.activity_descriptor_queue_tests import ActivityDescriptorQueueTest

if __name__ == '__main__':
    unittest.main()

