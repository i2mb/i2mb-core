from i2mb.worlds import CompositeWorld


class Store(CompositeWorld):
    def __init__(self, **kwargs):
        kwargs["waiting_room"] = True
        CompositeWorld.__init__(self, **kwargs)

    def enter_world(self, n, idx=None, arriving_from=None):
        pass
