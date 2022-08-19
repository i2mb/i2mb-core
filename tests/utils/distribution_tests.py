from functools import partial

import numpy as np

from i2mb.utils.distributions import TemporalLinkedDistribution
from tests.i2mb_test_case import I2MBTestCase


class TestDistributions(I2MBTestCase):
    def test_temporal_linked_distribution(self):
        on_distribution = partial(np.random.randint, 0, 100)
        tld = TemporalLinkedDistribution(on_distribution, 2)

        # Multiple values
        on_time = tld.sample_on(10)
        self.assertEqual(len(on_time), 10)

        off_time = tld.sample_off()
        self.assertEqual(len(off_time), 10)

        self.assertEqualAll(off_time/2, on_time)

        # Single value
        on_time = tld.sample_on()
        self.assertIsInstance(on_time, int)

        off_time = tld.sample_off()
        self.assertIsInstance(off_time, int)
        self.assertEqual(off_time / 2, on_time)




