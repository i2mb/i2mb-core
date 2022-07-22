from unittest import TestCase

__unittest = True


class I2MBTestCase(TestCase):
    """Extension of teh TEstCase class to include numpy array comparisons"""
    def assertTrueAny(self, test_value, msg=""):
        if not test_value.any():
            self.fail(f"Failed (test_value).any() test: {test_value[:15]}\n{msg}")

    def assertTrueAll(self, test_value, msg=""):
        if not test_value.all():  # noqa
            self.fail(f"Failed == .all() test: {test_value[:15]}\n{msg}")

    def assertEqualAny(self, value, test_value, msg=""):
        if not (value == test_value).any():
            self.fail(f"Failed (value == test_value).any() test: {(value == test_value)[:15]}\n{msg}")

    def assertEqualAll(self, value, test_value, msg=""):
        if not (value == test_value).all():  # noqa
            self.fail(f"Failed == .all() test: {(value == test_value)[:15]}\n{msg}")
            
    def assertGreaterAll(self, value, test_value, msg=""):
        if not (value > test_value).all():  # noqa
            self.fail(f"Failed > all() test: {(value > test_value)[:15]}\n{msg}")
            
    def assertGreaterAny(self, value, test_value, msg=""):
        if not (value > test_value).any():  # noqa
            self.fail(f"Failed > .any() test: {(value > test_value)[:15]}\n{msg}")

    def assertGreaterEqualAll(self, value, test_value, msg=""):
        if not (value >= test_value).all():  # noqa
            self.fail(f"Failed >= all() test: {(value >= test_value)[:15]}\n{msg}")

    def assertGreaterEqualAny(self, value, test_value, msg=""):
        if not (value >= test_value).any():  # noqa
            self.fail(f"Failed >= .any() test: {(value >= test_value)[:15]}\n{msg}")

    def assertLessAll(self, value, test_value, msg=""):
        if not (value < test_value).all():  # noqa
            self.fail(f"Failed < all() test: {(value < test_value)[:15]}\n{msg}")

    def assertLessAny(self, value, test_value, msg=""):
        if not (value < test_value).any():  # noqa
            self.fail(f"Failed < .any() test: {(value < test_value)[:15]}\n{msg}")

    def assertLessEqualAll(self, value, test_value, msg=""):
        if not (value <= test_value).all():  # noqa
            self.fail(f"Failed <= all() test: {(value <= test_value)[:15]}\n{msg}")

    def assertLessEqualAny(self, value, test_value, msg=""):
        if not (value <= test_value).any():  # noqa
            self.fail(f"Failed <= .any() test: {(value <= test_value)[:15]}\n{msg}")

    def assertNotEqualAny(self, value, test_value, msg=""):
        if not (value != test_value).any():
            self.fail(f"Failed (value != test_value).any() test: {(value != test_value)[:15]}\n{msg}")

    def assertNotEqualAll(self, value, test_value, msg=""):
        if not (value != test_value).all():  # noqa
            self.fail(f"Failed != .all() test: {(value != test_value)[:15]}\n{msg}")
