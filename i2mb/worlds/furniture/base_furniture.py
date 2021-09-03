from i2mb.worlds._area import Area


class BaseFurniture(Area):
    def __init__(self, height, width, origin=None, rotation=0, scale=1):
        self.equipment = []
        super().__init__(height=height, width=width, origin=origin, rotation=rotation, scale=scale,
                         subareas=self.equipment)

    def get_activity_position(self, origin=(0, 0), pos_id=None):
        return self.origin + self.dims / 2

    def add_equipment(self, equipment):
        self.points.extend([p for f in equipment for p in [f.origin, f.opposite]])
        self.equipment.extend(equipment)







