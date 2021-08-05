from .base_motion import Motion


class BeaconRoutines(Motion):
    def __init__(self, world, population):
        super().__init__(world, population)
