from unittest import TestCase

from i2mb.utils.collections import ConstrainedDict


class TestConstrainedDict(TestCase):
    def setUp(self):
        constraints = {"a", "b"}
        self.a = ConstrainedDict(constraints)

    def test_setting_item(self):
        self.a["c"] = 1
        self.assertDictEqual(self.a, {"c": 1})  # add assertion here

    def test_setting_item_in_constraint(self):
        self.assertRaises(KeyError, self.a.__setitem__, "a", 1)

    def test_message(self):
        self.a.msg = "Test message for key {}"
        try:
            self.a["b"] = 1
        except KeyError as e:
            self.assertEqual(str(e), "'Test message for key b'")


