from masskrug.worlds import CompositeWorld, Bar


class Apartment(CompositeWorld):
    def __init__(self, **kwargs):
        self.bar = Bar(shape="U")
        self.sleeping_rooms = [SleepingRoom() for i in range(3)]

        # Room Personalization
        sr1_dim = self.sleeping_rooms[0].dims
        self.sleeping_rooms[1].origin = (0, sr1_dim[1])
        self.bar.origin = (sr1_dim[0], 0)


class SleepingRoom(CompositeWorld):
    def __init__(self, num_beds=1, bed_sizes=None, rotation=0, **kwargs):
        super().__init__(**kwargs)
        self.beds = [Bed(bed_sizes[i]) for i in range(num_beds)]

    @CompositeWorld.dims.setter
    def dims(self, v):
        CompositeWorld.dims.fset(self, v)
        for bed in self.beds:
            bed.offset += [v[0] / 2, 0]
