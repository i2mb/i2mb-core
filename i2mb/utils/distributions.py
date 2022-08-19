from numbers import Number

from i2mb.utils.functions import callable_number


class TemporalLinkedDistribution:
    """Provides an on - off sampling method where the on sample is drawn form a provided distribution and the off sample
    is computed based on the original distribution.
    Note: The sample_on, sample_off methods need to be called sequentially starting by the sample_on method."""
    def __init__(self, on_dist, factor):
        self.on_dist = on_dist

        if isinstance(factor, Number):
            factor = callable_number(factor)

        self.off_factor = factor
        self.__on_time = 1
        self.__size = None

    def sample_on(self, size=None):
        self.__on_time = self.on_dist(size)
        self.__size = size
        return self.__on_time

    def sample_off(self, size=None):
        return self.__on_time * self.off_factor(self.__size)





