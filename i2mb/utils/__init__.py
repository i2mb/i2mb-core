from numpy import int64

from .cache_manager import CacheManager
from .time import SimulationTime


class Enumerator:
    """
    Used to create explicit enumerations, without having to hard code the enum value. Simply instantiate the class,
    and call :func:`auto` to assign the corresponding enumeration value.

    Example:
        >>> my_enumerator = Enumerator()
        >>> a = my_enumerator.auto()
        >>> b = my_enumerator.auto()
        >>> c = my_enumerator.auto()
        >>> d = my_enumerator.auto()
        >>> print(a, b, c, d)
        >>> 0, 1, 2, 3

    """
    def __init__(self):
        self.__counter = 0

    def auto(self):
        current = self.__counter
        self.__counter += 1
        return current


cache_manager = CacheManager()

global_time = SimulationTime()
int_inf = int64(0x4FFFFFFFFFFFFFFF)
