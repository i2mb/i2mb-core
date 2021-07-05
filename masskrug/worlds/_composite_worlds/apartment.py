from random import randint, choice
import numpy as np
from matplotlib.patches import Rectangle
from masskrug.worlds import CompositeWorld
from masskrug.worlds import Bath, BedRoom, LivingRoom, Kitchen, DiningRoom, Corridor, BlankSpace
from masskrug.worlds.furniture.door import DOOR_WIDTH

"""
    :param num_residents: Number of residents, values not between 1 and 6 will be ignored
    :type num_residents: int
    :param rotation: Angle of counterclockwise rotation of the apartment. Measured in degrees.
        Values other than 0, 90, 180 and 270 are ignored.
    :type rotation: int, optional
    :param scale: percentage value for apartment scaling
    :type scale: float, optional 
"""


class Apartment(CompositeWorld):
    def __init__(self, num_residents=None, rotation=0, scale=1, dims=(15, 7), floor_number=0, **kwargs):
        super().__init__(dims=dims, **kwargs)
        self.dims = self.dims * scale
        self.rotation = rotation
        if num_residents is None:
            num_residents = 1
        num_residents = (min(6, max(1, num_residents)))
        self.num_residents = num_residents
        if rotation != 0 and rotation != 90 and rotation != 180 and rotation != 270:
            print("not a valid rotation")
            self.__rotation = 0

        # random value for shower or bathtub
        guest = randint(0, 1)
        # random value for kitchen outline
        kitchen = choice(["U", "I"])  # "L", "I"])

        self.floor_number = floor_number

        # rooms have relativ positions in the apartment
        # apartment entry on right side
        if self.rotation == 0:
            self.livingroom = LivingRoom(num_seats=num_residents + 2, origin=(0, 0), scale=scale,
                                         rotation=(self.rotation - 90) % 360)
            self.corridor = Corridor(
                origin=(self.livingroom.dims[0], (self.dims[1] - scale) / 2),
                rotation=(self.rotation - 90) % 360, scale=scale)
            bedroom_width = (self.dims[0] - self.livingroom.dims[0]) / (num_residents // 2 + num_residents % 2)

            self.bedroom = [BedRoom(num_beds=(s + 1) * 2 - num_residents % 2, rotation=self.rotation,
                                    origin=[s * bedroom_width + self.livingroom.dims[0],
                                            0],
                                    dims=[bedroom_width / scale, (self.dims[1] - scale) / (2. * scale)], scale=scale)
                            for s in range(num_residents // 2 + num_residents % 2)]

            self.diningroom = DiningRoom(origin=(
                (self.livingroom.dims[0]), (self.bedroom[0].dims[1] + 1 * scale)),
                rotation=(self.rotation + 180) % 360, scale=scale)
            self.kitchen = Kitchen(outline=kitchen,
                                   origin=(
                                       self.diningroom.origin[0] + self.diningroom.dims[0], self.diningroom.origin[1]),
                                   rotation=(self.rotation + 180) % 360, scale=scale)
            self.bath = Bath(guest=guest,
                             origin=(self.kitchen.origin[0] + self.kitchen.dims[0], self.kitchen.origin[1]),
                             rotation=(self.rotation + 180) % 360, scale=scale)

            # entry points
            dining_exit = [self.diningroom.entry_point[0], self.corridor.dims[1] - DOOR_WIDTH / 4]
            kitchen_exit = [self.kitchen.entry_point[0] + self.diningroom.dims[0],
                            self.corridor.dims[1] - DOOR_WIDTH / 4]
            bath_exit = [self.bath.entry_point[0] + self.diningroom.dims[0] + self.kitchen.dims[0],
                         self.corridor.dims[1] - DOOR_WIDTH / 4]
            living_exit = [DOOR_WIDTH / 4, self.corridor.dims[1] / 2]
            bed_exit = []
            for it, b in enumerate(self.bedroom):
                bed_exit += [[b.entry_point[0] + b.dims[0] * it, DOOR_WIDTH / 4]]

        # apartment entry on left side
        if self.rotation == 180:
            self.livingroom = LivingRoom(num_seats=num_residents + 2, origin=(0, 0), scale=scale,
                                         rotation=(self.rotation - 90) % 360)
            self.livingroom.origin = (self.dims[0] - self.livingroom.dims[0], 0)

            bedroom_width = (self.dims[0] - self.livingroom.dims[0]) / (num_residents // 2 + num_residents % 2)
            self.corridor = Corridor(origin=(0, (self.dims[1] - scale) / 2),
                                     rotation=(self.rotation - 90) % 360, scale=scale)

            self.bedroom = [BedRoom(num_beds=(s + 1) * 2 - num_residents % 2, rotation=(self.rotation + 180) % 360,
                                    origin=[s * bedroom_width,
                                            0],
                                    dims=[bedroom_width / scale, (self.dims[1] - scale) / (2 * scale)], scale=scale)
                            for s in range(num_residents // 2 + num_residents % 2)]

            self.bath = Bath(guest=guest, origin=(0, self.bedroom[0].dims[1] + scale),
                             rotation=self.rotation, scale=scale)
            self.kitchen = Kitchen(outline=kitchen,
                                   origin=(
                                       self.bath.origin[0] + self.bath.dims[0], self.bath.origin[1]),
                                   rotation=self.rotation, scale=scale)
            self.diningroom = DiningRoom(origin=(
                self.kitchen.dims[0] + self.kitchen.origin[0], self.kitchen.origin[1]),
                rotation=self.rotation, scale=scale)

            # entry points
            dining_exit = [self.diningroom.entry_point[0] + self.kitchen.dims[0] + self.bath.dims[0],
                           self.corridor.dims[1] - DOOR_WIDTH / 4]
            kitchen_exit = [self.kitchen.entry_point[0] + self.bath.dims[0],
                            self.corridor.dims[1] - DOOR_WIDTH / 4]
            bath_exit = [self.bath.entry_point[0], self.corridor.dims[1] - DOOR_WIDTH / 4]
            living_exit = [self.corridor.dims[0] - DOOR_WIDTH / 4, self.corridor.dims[1] / 2]
            bed_exit = []
            for it, b in enumerate(self.bedroom):
                bed_exit += [[b.entry_point[0] + b.dims[0] * it, DOOR_WIDTH / 4]]

        # apartment entry on upper side
        if self.rotation == 90:
            self.dims[0], self.dims[1] = self.dims[1], self.dims[0]
            self.livingroom = LivingRoom(num_seats=num_residents + 2, scale=scale, origin=(0, 0),
                                         rotation=(self.rotation - 90) % 360)

            self.corridor = Corridor(origin=((self.dims[0] - scale) / 2, self.livingroom.dims[1]),
                                     rotation=(self.rotation - 90) % 360, scale=scale)

            bedroom_width = (self.dims[1] - self.livingroom.dims[1]) / (num_residents // 2 + num_residents % 2)
            self.bedroom = [BedRoom(num_beds=(s + 1) * 2 - num_residents % 2, rotation=(self.rotation + 180) % 360,
                                    origin=[0, s * bedroom_width + self.livingroom.dims[1]],
                                    dims=[bedroom_width / scale, (self.dims[0] - scale) / (2. * scale)], scale=scale)
                            for s in range(num_residents // 2 + num_residents % 2)]

            self.diningroom = DiningRoom(origin=((self.bedroom[0].dims[0] + 1 * scale), (self.livingroom.dims[1])),
                                         rotation=self.rotation, scale=scale)

            self.kitchen = Kitchen(
                origin=(self.diningroom.origin[0], self.diningroom.origin[1] + self.diningroom.dims[1]),
                rotation=self.rotation, scale=scale, outline=kitchen)

            self.bath = Bath(origin=(self.kitchen.origin[0], self.kitchen.origin[1] + self.kitchen.dims[1]),
                             rotation=self.rotation, scale=scale, guest=guest)

            # entry points
            dining_exit = [self.corridor.dims[0] - DOOR_WIDTH / 4, self.diningroom.entry_point[1]]
            kitchen_exit = [self.corridor.dims[0] - DOOR_WIDTH / 4,
                            self.diningroom.dims[1] + self.kitchen.entry_point[1]]
            bath_exit = [self.corridor.dims[0] - DOOR_WIDTH / 4,
                         self.kitchen.dims[1] + self.diningroom.dims[1] + self.bath.entry_point[1]]
            living_exit = [self.corridor.dims[0] / 2, DOOR_WIDTH / 4]
            bed_exit = []
            for it, b in enumerate(self.bedroom):
                bed_exit += [[DOOR_WIDTH / 4, b.entry_point[1] + b.dims[1] * it]]

        # apartment entry on lower side
        if self.rotation == 270:
            self.dims[0], self.dims[1] = self.dims[1], self.dims[0]

            self.livingroom = LivingRoom(num_seats=num_residents + 2, origin=(0, 0), scale=scale,
                                         rotation=(self.rotation - 90) % 360)
            self.livingroom.origin = (0, self.dims[1] - self.livingroom.dims[1])

            self.corridor = Corridor(origin=((self.dims[0] - scale) / 2, 0),
                                     rotation=(self.rotation - 90) % 360, scale=scale)

            bedroom_width = (self.dims[1] - self.livingroom.dims[1]) / (num_residents // 2 + num_residents % 2)
            self.bedroom = [BedRoom(num_beds=(s + 1) * 2 - num_residents % 2, rotation=self.rotation,
                                    origin=[0, s * bedroom_width],
                                    dims=[bedroom_width / scale, (self.dims[0] - scale) / (2 * scale)], scale=scale)
                            for s in range(num_residents // 2 + num_residents % 2)]

            self.bath = Bath(guest=guest, origin=(self.bedroom[0].dims[0] + scale, 0),
                             rotation=(self.rotation + 180) % 360, scale=scale)

            self.kitchen = Kitchen(outline=kitchen,
                                   origin=(self.bath.origin[0], self.bath.origin[1] + self.bath.dims[1]),
                                   rotation=(self.rotation + 180) % 360, scale=scale)

            self.diningroom = DiningRoom(origin=(self.kitchen.origin[0], self.kitchen.origin[1] + self.kitchen.dims[1]),
                                         rotation=(self.rotation + 180) % 360, scale=scale)

            # enrty points
            dining_exit = [self.corridor.dims[0] - DOOR_WIDTH / 4,
                           self.diningroom.entry_point[1] + self.kitchen.dims[1] + self.bath.dims[1]]
            kitchen_exit = [self.corridor.dims[0] - DOOR_WIDTH / 4,
                            self.bath.dims[1] + self.kitchen.entry_point[1]]
            bath_exit = [self.corridor.dims[0] - DOOR_WIDTH / 4, self.bath.entry_point[1]]
            living_exit = [self.corridor.dims[0] / 2, self.corridor.dims[1] - DOOR_WIDTH / 4]
            bed_exit = []
            for it, b in enumerate(self.bedroom):
                bed_exit += [[DOOR_WIDTH / 4, b.entry_point[1] + b.dims[1] * it]]

        self.corridor.set_room_entries([dining_exit, kitchen_exit, bath_exit, living_exit],
                                       [id(self.diningroom), id(self.kitchen), id(self.bath), id(self.livingroom)])
        self.corridor.set_room_entries(bed_exit, [id(b) for b in self.bedroom])
        # add regions
        self.add_regions([self.corridor, self.kitchen, self.livingroom, self.diningroom, self.bath])

        self.add_regions(b for b in self.bedroom)

    def add_regions(self, regions):
        self.regions.extend(regions)

        region_origins = list(self.region_origins)
        region_dimensions = list(self.region_dimensions)
        region_origins.extend([r.origin for r in regions])
        region_dimensions.extend([r.dims for r in regions])

        self.region_origins = np.array(region_origins)
        self.region_dimensions = np.array(region_dimensions)

    def draw_world(self, ax=None, origin=(0, 0), **kwargs):
        bbox = kwargs.get("bbox", False)
        self._draw_world(ax, origin=origin, **kwargs)
        for region in self.regions:
            region.draw_world(ax=ax, origin=origin + self.origin, **kwargs)

    def _draw_world(self, ax=None, bbox=False, origin=(0, 0)):
        abs_origin = self.origin + origin
        ax.add_patch(Rectangle(abs_origin, *self.dims, fill=False, linewidth=1.2, edgecolor='gray'))

    def enter_world(self, n, idx=None, locations=None):
        if hasattr(self.population, "motion_mask"):
            self.population.motion_mask[:] = False
        return [self.corridor.entry_point] * n

    def check_positions(self, mask):
        for r in self.regions:
            # add support for sub regions
            if type(r) is not BlankSpace and len(r.regions) > 0:
                for rooms in r.regions:
                    if rooms.is_empty():
                        continue
                    r_mask = mask[rooms.population.index]
                    rooms.check_positions(r_mask, r.origin)

            if r.is_empty():
                continue
            r_mask = mask[r.population.index]
            r.check_positions(r_mask)

        if self.is_empty():
            return

        positions = self.population.position[self.location == self]
        mask = mask[self.location == self]
        check_top = mask.ravel() & (positions[:, 0] > self.dims[0])
        check_right = mask.ravel() & (positions[:, 1] > self.dims[1])
        check_left = mask.ravel() & (positions[:, 0] < 0)
        check_bottom = mask.ravel() & (positions[:, 1] < 0)

        positions[check_top, 0] = self.dims[0]
        positions[check_right, 1] = self.dims[1]
        positions[check_left, 0] = 0
        positions[check_bottom, 1] = 0
        self.population.position[self.location == self] = positions
        for lm in self.landmarks:
            lm.remove_overlap()
