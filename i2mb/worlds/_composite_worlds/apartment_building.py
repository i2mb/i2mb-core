from random import randint
from matplotlib.patches import Rectangle

from i2mb.worlds import CompositeWorld
from i2mb.worlds import Apartment, Corridor, Lift, Stairs

"""
    :param num_floors: Number of floors in an apartment building
    :type num_floors: int, optional
    :param num_apartments: Number of apartments per floor 
    :type num_apartments: int, optional
    :param scale: percentage value for apartment building scaling
    :type scale: float, optional 
"""


class ApartmentBuilding(CompositeWorld):
    def __init__(self, num_apartments=6, num_floors=2, scale=1, **kwargs):
        super().__init__(**kwargs)
        # self.num_floors = max(2, min(6, num_floors))
        self.num_floors = num_floors
        self.num_apartments = max(2, min(8, num_apartments))
        apartment_dims = (15, 7)
        corridor_dims = (1.5, apartment_dims[1] * self.num_apartments)
        height = self.num_floors * (apartment_dims[0] + corridor_dims[0])
        self.apartments = []
        self.corridor = []
        self.dims = (apartment_dims[1] * self.num_apartments + 9) * scale, height * scale

        self.stairs = Stairs(num_floors=self.num_floors, scale=scale, dims=(6, height),
                             origin=(self.num_apartments * apartment_dims[1] * scale, 0))
        for i in range(self.num_floors):
            self.apartments += ([Apartment(num_residents=randint(1, 6), rotation=270, dims=apartment_dims, scale=scale,
                                           origin=(s * apartment_dims[1] * scale,
                                                   ((i * (apartment_dims[0] + corridor_dims[0]) +
                                                     corridor_dims[0]) * scale)), floor_number=i)
                                 for s in range(self.num_apartments)])
            self.corridor += ([Corridor(origin=(0, (i * (apartment_dims[0] + corridor_dims[0])) * scale),
                                        public=True, floor_number=i, dims=corridor_dims, scale=scale, rotation=270)])
            apartment_entry = [[apartment_dims[1] * c + (apartment_dims[1] / 2), corridor_dims[0] - 0.2] for c in
                               range(self.num_apartments)]
            apartment_id = [id(a.corridor) for a in
                            self.apartments[i * self.num_apartments: (i + 1) * self.num_apartments]]
            self.corridor[i].set_room_entries(apartment_entry, apartment_id)
            self.corridor[i].set_room_entries([self.corridor[i].entry_point], [id(self.stairs)])

        for a in self.apartments:
            a.corridor.set_room_entries([a.corridor.entry_point], [id(self.corridor[a.floor_number])])

        corridor_entries = [[0.2, (i * (apartment_dims[0] + corridor_dims[0])) + corridor_dims[0] / 2] for i in
                            range(self.num_floors)]

        corridor_ids = [id(c) for c in self.corridor]

        self.stairs.set_room_entries(corridor_entries, corridor_ids)

        self.lift = Lift(origin=(self.stairs.origin[0] + self.stairs.dims[0], self.stairs.origin[1]),
                         scale=scale, width=3, height=height)
        self.stairs.set_room_entries([self.stairs.entry_point], [id(self.lift)])

        self.add_regions([self.stairs, self.lift])
        self.add_regions(a for a in self.apartments)
        self.add_regions(c for c in self.corridor)

        self.num_residents = sum([a.num_residents for a in self.apartments])

    def draw_world(self, ax=None, origin=(0, 0), **kwargs):
        bbox = kwargs.get("bbox", False)
        self._draw_world(ax, origin=origin, **kwargs)
        for region in self.regions:
            region.draw_world(ax=ax, origin=origin + self.origin, **kwargs)

    def _draw_world(self, ax=None, bbox=False, origin=(0, 0), **kwargs):
        ax.add_patch(Rectangle(self.origin + origin, *self.dims, fill=False, linewidth=1.2, edgecolor="gray"))
