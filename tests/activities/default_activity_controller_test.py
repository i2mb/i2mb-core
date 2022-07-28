from unittest import TestCase, skip

from i2mb.activities import ActivityDescriptorProperties
from i2mb.activities.activity_descriptors import Rest, Work
from i2mb.activities.activity_manager import ActivityManager
from i2mb.activities.atomic_activities import Sleep
from i2mb.activities.base_activity import ActivityNone
from i2mb.activities.base_activity_descriptor import ActivityDescriptor
from i2mb.activities.controllers.default_activity_controller import DefaultActivityController
from i2mb.engine.agents import AgentList
from i2mb.utils import global_time, time
from i2mb.worlds import Apartment
from tests.i2mb_test_case import I2MBTestCase
from tests.world_tester import WorldBuilder


class TestDefaultActivityController(I2MBTestCase):
    def walk_modules(self, num_steps=1):
        for i in range(num_steps):
            global_time.set_sim_time(i)
            self.activity_manager.pre_step(i)
            self.default_activity_controller.step(i)
            self.activity_manager.step(i)
            self.activity_manager.post_step(i)

    def setUp(self) -> None:
        self.population_size = 10
        self.population = AgentList(self.population_size)
        global_time.set_sim_time(0)
        self.world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=6),
                             sim_duration=global_time.make_time(day=4), no_gui=True)

        relocator = self.world.relocator
        self.activity_manager = ActivityManager(self.population, relocator=relocator)
        self.activity_manager.register_location_activities()
        self.default_activity_controller = DefaultActivityController(self.population, self.activity_manager)
        # self.register_available_location_activities()

    def test_default_activity_mapping(self):
        # This is a feature of teh apartment, but I guess it makes sense to test it here (?)
        default_activity = self.world.universe.regions[0].default_activity
        self.assertTrue(isinstance(default_activity, Rest))

        for region in self.world.universe.regions[0].regions:
            self.assertEqual(default_activity, region.default_activity)

    def test_default_activity_on_step(self):
        self.assertTrue((self.activity_manager.current_activity == -1).all())

        # apply local default activities
        region_1 = self.world.universe.regions[0].default_activity  # type: Rest
        region_2 = self.world.universe.regions[1].default_activity
        self.set_default_activity_description(region_1, region_2)

        self.walk_modules(5)
        self.assertGreaterAny(self.activity_manager.current_activity, -1)
        self.assertEqualAll(self.activity_manager.current_activity, region_1.activity_class.id)

    def set_default_activity_description(self, region_1, region_2):
        self.default_activity_controller.current_default_activity_descriptor[:5] = \
            region_1.create_specs().specifications
        self.default_activity_controller.current_default_activity_descriptor[5:] = \
            region_2.create_specs().specifications

    def test_default_activity_restart(self):
        # Walk to stage deafult activity
        self.walk_modules(5)
        self.assertEqualAll(self.activity_manager.current_activity, 0)
        self.activity_manager.stop_activities(time(), self.population.index)
        descriptors = Rest(location=self.world.universe).create_specs(size=self.population_size).specifications
        descriptors[:, ActivityDescriptorProperties.location_ix] = 0
        self.activity_manager.stage_activity(descriptors,
                                             self.population.index)
        self.walk_modules(5)
        self.assertEqualAll(self.activity_manager.current_activity, Rest().activity_class.id,
                            msg=f"{self.activity_manager.current_activity}")

        self.walk_modules(5)
        self.assertEqualAll(self.activity_manager.current_activity, Rest().activity_class.id)

        self.activity_manager.stop_activities(time(), self.population.index)
        self.walk_modules(5)
        self.assertEqualAll(self.activity_manager.current_activity, 0)

    def test_default_activity_on_enter(self):
        self.default_activity_controller.post_init()
        self.world.relocator.move_agents(self.population.index, self.world.universe.regions[2])
        default_activity_a = self.world.universe.regions[0].default_activity  # type: ActivityDescriptor
        default_activity_b = self.world.universe.regions[1].default_activity  # type: ActivityDescriptor

        self.world.relocator.move_agents(self.population.index[:5], self.world.universe.regions[0])
        self.world.relocator.move_agents(self.population.index[5:], self.world.universe.regions[1])
        self.assertEqualAll(self.default_activity_controller.current_default_activity,
                            default_activity_b.activity_class.id)

        self.walk_modules(5)
        self.assertGreaterAny(self.activity_manager.current_activity, -1)
        self.assertEqualAll(self.activity_manager.current_activity, default_activity_b.activity_class.id)

        self.assertEqualAll(default_activity_a.location,  self.population[:5].location,
                            msg=f"{self.population.location[:5]}")
        self.assertEqualAll(default_activity_b.location, self.population[5:].location,
                            msg=f"{self.population.location[5:]}")

    def test_default_activity_on_exit(self):
        self.default_activity_controller.post_init()
        default_activity_a = self.world.universe.regions[0].default_activity  # type: ActivityDescriptor
        default_activity_b = self.world.universe.regions[1].default_activity  # type: ActivityDescriptor
        self.set_default_activity_description(default_activity_a, default_activity_b)
        self.walk_modules(5)
        self.assertEqualAll(self.activity_manager.current_activity, default_activity_b.activity_class.id)

        self.world.relocator.move_agents(self.population.index, self.world.universe.regions[2])
        self.assertEqualAll(self.default_activity_controller.current_default_activity, ActivityNone.id,
                            msg=f"{self.default_activity_controller.current_default_activity}")

    @skip("Please implement")
    def test_respect_staged_activities(self):
        """This was found when testing integration with sleep. Make sure that when an activity is in the staging
        area, default activity does not overwrite. Both on_entry and in step methods."""
        ...













