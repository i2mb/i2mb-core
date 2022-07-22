import numpy as np

from i2mb.engine.agents import AgentList
from tests.i2mb_test_case import I2MBTestCase


class TestAgentList(I2MBTestCase):
    def setUp(self) -> None:
        self.population = AgentList(10)
        self.property1 = np.zeros((10, 1))
        self.property2 = np.zeros((10, 2))

        self.population.add_property("property1", self.property1)
        self.population.add_property("property2", self.property2)

    def test_setting_scalar(self):
        idx = [2, 3, 5, 6]

        # Set on the original buffer
        self.property1[idx, :] = 1
        self.assertEqualAll(self.population.property1, self.property1, msg=f"{self.population.property1}")

        self.property2[idx, 1] = 2
        self.assertEqualAll(self.population.property2, self.property2, msg=f"{self.population.property2}")

        # Set on the AgentList
        self.population.property1[idx] = 0
        self.assertEqualAll(self.population.property1, self.property1, msg=f"{self.population.property1}")

        self.population.property2[idx] = 0
        self.assertEqualAll(self.population.property2, self.property2, msg=f"{self.population.property2}")

    def test_setting_scalar_on_a_view(self):
        idx = [2, 3, 5, 6]
        view = self.population[idx]

        view.property1[[2, 3]] = 3
        self.assertEqualAny(self.population.property1[5:7], 3, msg=f"{self.population.property1}")
        self.assertEqualAll(self.population.property1, self.property1, msg=f"{self.population.property1}")

        view.property2[[2, 3], 1] = 3
        self.assertEqualAny(self.population.property2[5:7], [0, 3], msg=f"{self.population.property1}")
        self.assertEqualAll(self.population.property2, self.property2, msg=f"{self.population.property2}")
