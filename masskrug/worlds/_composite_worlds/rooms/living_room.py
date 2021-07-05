from ._room import BaseRoom
from masskrug.utils import global_time
from functools import partial
from collections import deque
import numpy as np

from masskrug.worlds.furniture.furniture_living.seats import Sofa
from masskrug.worlds.furniture.furniture_living.seats import Armchair
from masskrug.worlds.furniture.furniture_living.couch_table import CouchTable

"""
    :param num_seats: Number of seats, values not between 1 and 6 will be ignored
    :type num_seats: int, optional
"""


class LivingRoom(BaseRoom):
    def __init__(self, num_seats=2, dims=(7, 4.5), scale=1, **kwargs):
        super().__init__(dims=dims, scale=scale, **kwargs)
        width, height = self.dims[0], self.dims[1]
        offset = [0, 0, height, width, 0]
        x, y = self.origin[0], self.origin[1]
        rot = np.radians(self.rotation)
        num_seats = max(1, min(num_seats, 6))

        self.sofa = [Sofa(self.rotation, scale=scale) for s in range(num_seats // 2)]
        self.armchair = [Armchair(self.rotation, scale=scale) for s in range(num_seats % 2)]

        if num_seats == 1:
            sofa_width = self.armchair[0].width
            sofa_length = self.armchair[0].length
        else:
            sofa_width = self.sofa[0].width
            sofa_length = self.sofa[0].length
        sofa_rotation = np.radians(self.rotation - 90)

        sofa_offset = [np.cos(rot) * 0.5 * (width - sofa_width) - np.sin(rot) * 0.1 * height -
                       np.cos(sofa_rotation) * (sofa_width + sofa_length) + np.sin(sofa_rotation) * (
                               sofa_length + height * 0.05),
                       np.sin(rot) * 0.5 * (width - sofa_width) + np.cos(rot) * 0.1 * height -
                       np.sin(sofa_rotation) * (sofa_width + sofa_length) - np.cos(sofa_rotation) * (
                               sofa_length + height * 0.05)]
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
        # sofa position
        i = -1
        for s in self.sofa:
            s.rotation += 90 * i
            rad = np.radians(s.rotation)
            s.origin = [sofa_offset[0] + offset[int(self.rotation / 90) + 1],
                        sofa_offset[1] + offset[int(self.rotation / 90)]]
            sofa_offset[0] += np.cos(rad) * (s.width + s.length + width * 0.05) - \
                              np.sin(rad) * (s.length + height * 0.05)
            sofa_offset[1] += np.sin(rad) * (s.width + s.length + width * 0.05) + \
                              np.cos(rad) * (s.length + height * 0.05)
            s.set_sitting_position()
            s.set_sleeping_position()
            self.sitting_pos.extend(s.get_sitting_position())
            self.target_pos.extend(s.get_sitting_target())
            i += 1

        # armchair position
        for a in self.armchair:
            a.rotation += 90 * i
            rad = np.radians(a.rotation)
            a.origin = [sofa_offset[0] + offset[int(self.rotation / 90) + 1],
                        sofa_offset[1] + offset[int(self.rotation / 90)]]
            sofa_offset[0] += np.cos(rad) * (a.width + a.length + width * 0.05) - \
                              np.sin(rad) * (a.length + height * 0.05)
            sofa_offset[1] += np.sin(rad) * (a.width + a.length + width * 0.05) + \
                              np.cos(rad) * (a.length + height * 0.05)
            a.set_sitting_position()
            self.sitting_pos.extend(a.get_sitting_position())
            self.target_pos.extend(a.get_sitting_target())
            i += 1

        self.sitting_pos = np.array(self.sitting_pos)
        self.target_pos = np.array(self.target_pos)

        self.seats = []
        self.seats.extend(s for s in self.sofa)
        self.seats.extend(a for a in self.armchair)

        self.available_seats = deque(self.seats, )
        self.seat_assignment = {}

        # rotate room
        if self.rotation == 90 or self.rotation == 270:
            self.dims[0], self.dims[1] = self.dims[1], self.dims[0]

        self.furniture += self.sofa
        self.furniture += self.armchair
        # self.furniture += [self.table]

        self.furniture_origins = np.empty((len(self.furniture) - 1, 2))
        self.furniture_upper = np.empty((len(self.furniture) - 1, 2))
        self.get_furniture_grid()

    def sit_particles(self, idx):
        bool_idx = (self.population.index.reshape(-1, 1) == idx).any(axis=1)
        new_idx = np.arange(len(self.population))[bool_idx]
        num_seats = len(idx)
        num = 0
        start = 0
        for t in range(num_seats):
            if not self.available_seats:
                return

            seat = self.available_seats[0]
            end = seat.num_seats - seat.occupants
            seats = len(idx[start:start + end])
            num += seats

            self.population.target[new_idx[start:start + end]] = seat.get_sitting_target()[
                                                                 seat.occupants:seats + seat.occupants]
            self.population.sitting_position[new_idx[start:start + end]] = seat.get_sitting_position()[
                                                                           seat.occupants:seats + seat.occupants]
            self.seat_assignment.update(dict.fromkeys(idx[start:start + end], seat))
            seat.occupants += seats
            start += end

            if seat.occupants == seat.num_seats:
                x = self.available_seats.popleft()
            if num >= len(idx):
                break

    def stand_up_particle(self, idx):
        for ix in idx:
            seat = self.seat_assignment[ix]
            seat.occupants -= 1
            del self.seat_assignment[ix]
            in_queue = seat in self.available_seats
            if not in_queue:
                self.available_seats.append(seat)

    def exit_world(self, idx):
        bool_ix = self.population.find_indexes(idx)
        self.population.motion_mask[bool_ix] = True
        self.population.accumulated_sitting[bool_ix] = 0
        self.population.current_sitting_duration[bool_ix] = -np.inf
        self.population.next_sitting_time[bool_ix] = -np.inf
        sitting = self.population.is_sitting.ravel() & bool_ix
        if sitting.any():
            self.population.is_sitting[sitting] = False
            self.stand_up_particle(self.population.index[sitting])

    def step(self, t):
        if not hasattr(self, "population"):
            return
        if not self.population:
            return

        n = len(self.population)
        # update sitting duration
        sit_update = self.population.is_sitting.ravel()
        if sit_update.any():
            self.population.accumulated_sitting[sit_update] += 1

        acc_sitting = self.population.accumulated_sitting.ravel()
        cur_sitting = self.population.current_sitting_duration.ravel()
        enough_sitting = (acc_sitting > cur_sitting)
        enough_sitting = sit_update & enough_sitting

        if enough_sitting.any():
            self.population.is_sitting[enough_sitting] = False
            self.stand_up_particle(self.population.index[enough_sitting])
            self.population.motion_mask[enough_sitting] = True
            self.population.target[enough_sitting] = np.nan
            self.population.accumulated_sitting[enough_sitting] = 0
            self.population.current_sitting_duration[enough_sitting] = -np.inf

        at_target = np.isclose(self.population.target, self.population.position)
        at_target = np.array([all(i) for i in at_target])
        at_target = at_target & self.population.motion_mask.ravel()

        if at_target.any():
            self.population.motion_mask[at_target] = False
            self.population.target[at_target] = self.population.sitting_position[at_target]

        next_sit = self.population.next_sitting_time.ravel()
        sit_again = (next_sit < t)

        if sit_again.any():
            # sit down for 70 to 90 minutes
            sit_duration = partial(np.random.normal, global_time.make_time(minutes=80),
                                   global_time.make_time(minutes=5))((sit_again.sum(), 1))

            self.population.current_sitting_duration[sit_again] = sit_duration

            has_target = np.ones((n, 2), dtype=bool)
            has_target[sit_again] = (self.population.target[sit_again] == self.entry_point)
            has_target = np.array([all(i) for i in has_target])
            has_no_target = ~has_target & ~self.population.is_sitting.ravel()

            if has_no_target.any():
                self.sit_particles(self.population.index[has_no_target])
                self.population.is_sitting[has_no_target] = True

            # walk around for 6 to 14 minutes
            next_time = partial(np.random.normal, global_time.make_time(minutes=10),
                                global_time.make_time(minutes=2))(
                (sit_again.sum(), 1))
            self.population.next_sitting_time[sit_again] = t + (next_time + sit_duration).astype(int)

        return
