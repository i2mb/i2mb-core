from numpy import int64

from .cache_manager import CacheManager
from .time import SimulationTime

cache_manager = CacheManager()

global_time = SimulationTime()
int_inf = int64(0x4FFFFFFFFFFFFFFF)
