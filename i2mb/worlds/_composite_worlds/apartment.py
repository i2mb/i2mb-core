from random import randint, choice

import numpy as np

from i2mb.worlds import Bathroom, BedRoom, LivingRoom, Kitchen, DiningRoom, Corridor
from i2mb.worlds import CompositeWorld

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
    def __init__(self, num_residents=None, rotation=0, dims=(15, 7), floor_number=0, **kwargs):
        super().__init__(dims=dims, rotation=rotation, **kwargs)

        kwargs.pop("population", None)
        if num_residents is None:
            num_residents = 1

        num_residents = (min(6, max(1, num_residents)))
        self.num_residents = num_residents

        # random value for shower or bathtub
        guest = randint(0, 1)
        # random value for kitchen outline
        kitchen = choice(["U", "L", "I"])  # "L", "I"])

        self.floor_number = floor_number

        scale = self.scale
        self.__build_living_room(num_residents, scale)
        self.__build_corridor(scale)
        self.__build_bedrooms(num_residents, scale)
        self.__build_bathroom(guest, scale)
        self.__build_kitchen(kitchen, scale)
        self.__build_dining_room(scale)

        self.register_rooms()
        self.__connect_rooms()
        self.fill_available_activities()

        self.beds = self.__get_beds()
        self.bed_assignment = []
        self.agent_bedroom = []
        self.default_activity = self.living_room.local_activities[0]

        for region in self.regions:
            region.default_activity = self.default_activity

        # People that actually live in this house
        self.inhabitants = None

    def register_rooms(self):
        # add regions
        self.add_regions([
            self.living_room,
            self.corridor,
            self.bathroom,
            self.kitchen,
            self.dining_room
        ])
        self.add_regions(self.bedrooms)

    def __build_dining_room(self, scale):
        self.dining_room = DiningRoom(origin=(self.living_room.width, self.kitchen.origin[1]),
                                      dims=[self.kitchen.height, self.kitchen.origin[0] - self.living_room.width],
                                      rotation=270, scale=scale)
        self.__create_entry_routes(self.dining_room)

    def __build_living_room(self, num_residents, scale):
        self.living_room = LivingRoom(num_seats=num_residents + 2, origin=(0, 0), scale=scale,
                                      rotation=0)
        self.__create_entry_routes(self.living_room)

    def __build_corridor(self, scale):
        length = max(self.dims) - min(self.living_room.dims)
        origin = [max(self.dims) - length, min(self.dims) / 2 - 0.5]
        self.corridor = Corridor(dims=(length, 1), origin=origin,
                                 rotation=0, public=False, scale=scale)
        self.__create_entry_routes(self.corridor)

    def __build_bedrooms(self, num_residents, scale):
        bedroom_width = (max(self.dims) - min(self.living_room.dims)) / (num_residents // 2 + num_residents % 2)
        bedroom_height = min(self.dims) / 2 - min(self.corridor.dims) / 2
        dims = [bedroom_height, bedroom_width]
        origin = np.array([min(self.living_room.dims), 0])
        self.bedrooms = [BedRoom(num_beds=(s + 1) * 2 - num_residents % 2, rotation=90,
                                 origin=np.array([bedroom_width, 0.]) * s + origin,
                                 dims=dims, scale=scale)
                         for s in range(num_residents // 2 + num_residents % 2)]

        for bedroom in self.bedrooms:
            self.__create_entry_routes(bedroom)

    def __build_bathroom(self, guest, scale):
        bathroom_width = 3
        bathroom_height = min(self.dims) / 2 - min(self.corridor.dims) / 2
        self.bathroom = Bathroom(guest=guest, origin=[self.width - bathroom_width, self.height - bathroom_height],
                                 rotation=270, scale=scale)

        self.__create_entry_routes(self.bathroom)

    def __build_kitchen(self, kitchen, scale):
        self.kitchen = Kitchen(outline=kitchen,
                               origin=(self.bathroom.origin[0] - 3.5, self.bathroom.origin[1]),
                               rotation=270, scale=scale)
        self.__create_entry_routes(self.kitchen)

    def __connect_rooms(self):
        corridor_origin = self.corridor.origin
        for room in self.regions:
            if room == self.corridor:
                continue

            entry = room.entry + room.origin - corridor_origin
            mask = (entry < 0) * (-2 * entry)
            entry += mask
            mask = (entry > self.corridor.dims) * (self.corridor.dims - (entry - self.corridor.dims))
            entry += mask
            self.corridor.set_room_entries(entry, room)

    def __create_entry_routes(self, region):
        region.entry_route = np.array(list(region.entry_route) + [self])

    # @property
    # def population(self):
    #     def get_population(region):
    #         if region.population is None:
    #             return np.array([])
    #
    #         return region.population.index.flatten()
    #
    #     return np.sort(np.fromiter((j for i in map(get_population, self.regions) for j in i), dtype=int))
    #
    # @population.setter
    # def population(self, v):
    #     return

    def get_entrance_sub_region(self):
        return self.corridor

    def assign_beds(self):
        self.beds = np.array([bed for bed in self.beds[:len(self.inhabitants)]])
        self.agent_bedroom = np.array([bed.parent for bed in self.beds])
        self.bed_assignment = self.inhabitants.index[:len(self.inhabitants)].copy().reshape(-1, 1)
        for room in self.bedrooms:
            room.assign_beds(self.bed_assignment[self.agent_bedroom == room])

    def move_home(self, population):
        self.inhabitants = population
        self.assign_beds()

    def __get_beds(self):
        beds = []
        for room in self.bedrooms:
            beds.extend(room.beds)

        return np.array(beds)

    def step(self, t):
        for r in self.regions:
            r.step(t)

    def fill_available_activities(self):
        for r in self.regions:
            self.available_activities.extend(r.local_activities)
            self.activity_types.update(type(act) for act in r.local_activities)

        for r in self.regions:
            r.available_activities = self.available_activities
            r.activity_types.update(self.activity_types)

    def prepare_entrance(self, idx, global_population):
        assert idx is not None

        if self.population is None:
            self.population = global_population[idx]
        else:
            self.population.add(idx)

        at_home_mask = (self.inhabitants.index.reshape(-1, 1) == idx).any(axis=1)
        self.inhabitants.at_home[at_home_mask] = True

    def exit_world(self, idx, global_population):
        # bool_idx = (self.population.reshape(-1, 1) == idx).any(axis=1)
        # print(bool_idx)
        self.population.remove(idx)

        at_home_mask = (self.inhabitants.index.reshape(-1, 1) == idx).any(axis=1)
        self.inhabitants.at_home[at_home_mask] = False
