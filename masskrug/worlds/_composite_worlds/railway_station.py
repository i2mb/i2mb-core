from masskrug.worlds import CompositeWorld


class RailwayStation(CompositeWorld):
    def __init__(self, **kwargs):
        CompositeWorld.__init__(self, **kwargs)
