import numpy as np

import i2mb.activities.activity_descriptors
from i2mb.worlds.furniture.furniture_living.seats import Armchair
from i2mb.worlds.furniture.furniture_living.seats import Sofa
from ._room import BaseRoom

# from i2mb.worlds.furniture.furniture_living.couch_table import CouchTable

"""
    :param num_seats: Number of seats, values not between 1 and 6 will be ignored
    :type num_seats: int, optional
"""


class LivingRoom(BaseRoom):
    def __init__(self, num_seats=2, dims=(4.5, 7), scale=1, **kwargs):
        super().__init__(dims=dims, scale=scale, **kwargs)
        self.num_seats = max(1, min(num_seats, 6))

        self.sofa = [Sofa(rotation=0, scale=scale) for s in range(num_seats // 2)]
        self.armchair = [Armchair(rotation=0, scale=scale) for s in range(num_seats % 2)]

        '''
        # couch table
        table_offset = [
            - np.sin(sofa_rotation) * (sofa_length * 1.2 + height * 0.05) + np.cos(sofa_rotation) * sofa_width * 0.05,
            np.cos(sofa_rotation) * (sofa_length * 1.2 + height * 0.05) + np.sin(sofa_rotation) * sofa_width * 0.05]

        table_rotation = self.rotation - 90

        if num_seats > 3:
            table_offset[0] += np.cos(sofa_rotation) * (sofa_width * 0.9) + np.sin(sofa_rotation) * (
                    sofa_length * 0.1)
            table_offset[1] += np.sin(sofa_rotation) * (sofa_width * 0.9) - np.cos(sofa_rotation) * (
                    sofa_length * 0.1)
            table_rotation += 90

        self.table = CouchTable(width=sofa_width * 0.9, length=sofa_length * 0.9, rotation=table_rotation,
                                origin=[sofa_offset[0] + table_offset[0] + offset[int(self.rotation / 90) + 1],
                                        sofa_offset[1] + table_offset[1] + offset[int(self.rotation / 90)]])
        '''
        self.sitting_pos = []
        self.target_pos = []

        # arranged in u-form
        self.arrange_furniture()

        self.arrange_sitting_positions_n_targets()
        # self.sitting_pos = np.array(self.sitting_pos)
        # self.target_pos = np.array(self.target_pos)

        self.seats = []
        self.seats.extend(s for s in self.sofa)
        self.seats.extend(a for a in self.armchair)

        self.add_furniture(self.sofa)
        self.add_furniture(self.armchair)
        # self.furniture += [self.table]

        self.furniture_origins = np.empty((len(self.furniture) - 1, 2))
        self.furniture_upper = np.empty((len(self.furniture) - 1, 2))
        self.get_furniture_grid()

        self.local_activities.extend([
            i2mb.activities.activity_descriptors.Rest(location=self, duration=45, blocks_for=1)
            # i2mb.activities.activity_descriptors.Rest(location=self, device=s, duration=20) for s in
            # self.seats for p in range(s.num_seats)
        ])

        # Seat management
        self.available_seats = np.array([pos for s in self.seats for pos in s.get_sitting_position()])
        self.seat_assignment = np.ones(len(self.available_seats)) * -1

    def arrange_furniture(self):
        if self.num_seats == 1:
            sofa_width = min(self.armchair[0].dims)
            sofa_length = max(self.armchair[0].dims)
        else:
            sofa_width = min(self.sofa[0].dims)
            sofa_length = max(self.sofa[0].dims)

        # sofa position
        width, height = self.dims
        furniture = self.armchair + self.sofa
        num_mid_sofas = len(furniture[2:])
        mid_height = max(num_mid_sofas * sofa_length + (num_mid_sofas - 1) * 0.2, sofa_length)
        sofa_angles = [0, 180]
        sofa_locations = [[0.20 + sofa_width + 0.20, height/2 - mid_height/2 - 0.20 - sofa_width],
                          [0.20 + sofa_width + 0.20, height/2 + mid_height/2 + 0.20]]

        for ix, f in enumerate(furniture[2:]):
            sofa_angles.append(270)
            sofa_locations.append([[0.20,  height/2 - mid_height/2 + (sofa_length + 0.2) * ix]])

        for loc, rot, f in zip(sofa_locations, sofa_angles, furniture):
            f.rotate(rot)
            f.origin = loc

    def arrange_sitting_positions_n_targets(self):
        for s in self.sofa:
            self.sitting_pos.extend(s.get_sitting_position())
            self.target_pos.extend(s.get_sitting_target())

        for a in self.armchair:
            self.sitting_pos.extend(a.sitting_pos)
            self.target_pos.extend(a.sitting_pos)

    def sit_particles(self, idx):
        bool_idx = self.population.find_indexes(idx)
        required_seats = len(idx)
        available_seats = (self.seat_assignment == -1)

        if available_seats.sum() > 0:
            if required_seats > available_seats.sum():
                required_seats = available_seats.sum()

            choose_seats = np.where(available_seats)[0][:required_seats]
            self.seat_assignment[choose_seats] = idx[:required_seats]

            choose_idx = np.where(bool_idx)[0][:required_seats]
            self.population.position[choose_idx] = self.available_seats[choose_seats]

        if required_seats < len(idx):
            choose_idx = np.where(~bool_idx)[0][:required_seats]
            self.population.position[choose_idx] = np.random.random((len(idx) - required_seats, 2)) * self.dims

    def stand_up_particle(self, idx):
        assigned_seats = (self.seat_assignment.reshape(-1, 1) == idx).any(axis=1)
        self.seat_assignment[assigned_seats] = -1
        bool_idx = self.population.find_indexes(idx)
        self.population.position[bool_idx] = self.dims / 2

    def start_activity(self, idx, activity_id):
        self.sit_particles(idx)

    def stop_activity(self, idx, descriptor_ids):
        self.stand_up_particle(idx)

    def step(self, t):
        if not hasattr(self, "population"):
            return
        if not self.population:
            return