import numpy as np

from i2mb.engine.agents import AgentList
from i2mb.engine.relocator import Relocator
from i2mb.motion.random_motion import RandomMotion
from i2mb.worlds import CompositeWorld, Apartment
from tests.i2mb_test_case import I2MBTestCase
from tests.world_tester import WorldBuilder


class TestRandomMotion(I2MBTestCase):
    def setUp(self) -> None:
        self.population = AgentList(10)
        self.world = CompositeWorld(regions=[CompositeWorld(dims=(10, 10))], population=self.population)
        self.relocator = Relocator(self.population, self.world)
        self.motion = RandomMotion(population=self.population)
        self.relocator.move_agents(self.population.index, self.world.regions[0])

    def verify_agent_in_region(self):
        dims = np.array([list(r.dims) for r in self.population.location])
        x_in_range = (0 <= self.population.position[:, 0]) & (self.population.position[:, 0] <= dims[:, 0])
        y_in_range = (0 <= self.population.position[:, 1]) & (self.population.position[:, 1] <= dims[:, 1])

        self.assertTrueAll(x_in_range, msg=self.population.position[:, 0])
        self.assertTrueAll(y_in_range, msg=(self.population.position[:, 1], dims[:, 1]))

    def test_update_return(self):
        self.population.motion_mask[:5] = False
        self.population.position[:] = np.random.random((len(self.population), 2)) * 10

        position = self.population.position.copy()
        updated = self.motion.update_positions(0)

        self.assertEqualAll(self.population.motion_mask.ravel(), updated)

        self.assertNotEqualAll(position[self.population.motion_mask.ravel()],
                               self.population.position[self.population.motion_mask.ravel()])

        self.assertEqualAll(position[~self.population.motion_mask.ravel(), :],
                            self.population.position[~self.population.motion_mask.ravel(), :])

    def test_containment(self):
        self.population.position[:, 0] = 0.25
        self.population.position[:, 1] = np.arange(10) * 10/11 + 10/11

        # Force agents to the wall
        self.motion.gravity_field = [-10, -10]

        for i in range(20):
            self.motion.step(i)
            self.verify_agent_in_region()

    def test_random_motion_of_partial_population(self):
        self.population.position[:, 0] = 0.25
        self.population.position[:, 1] = np.arange(10) * 10 / 11 + 10 / 11
        self.motion.motion_mask[:8] = False

        for i in range(20):
            self.motion.step(i)
            self.verify_agent_in_region()

    def test_random_motion_of_partial_population_complex_world(self):
        self.population = AgentList(10)
        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=5), no_gui=True,
                             sim_duration=20)

        world.relocator.move_agents(self.population.index[:], world.universe.regions[2])

        for i in range(20):
            self.motion.step(i)
            self.verify_agent_in_region()

        world.relocator.move_agents([0], world.universe.regions[0].regions[1])

        for i in range(20):
            self.motion.step(i)
            self.verify_agent_in_region()

    def test_random_motion_complex_world(self):
        self.population = AgentList(10)
        world = WorldBuilder(Apartment, population=self.population, world_kwargs=dict(num_residents=5), no_gui=True,
                             sim_duration=20)

        for i in range(40):
            world.engine.step()
            self.verify_agent_in_region()









