from ._room import BaseRoom
from i2mb.utils import global_time
from functools import partial
import numpy as np

from i2mb.worlds.furniture.bed import Bed

"""
    :param num_beds: Number of beds, values other than 1 and 2 are ignored
    :type num_beds: int, optional
"""


class BedRoom(BaseRoom):
    def __init__(self, num_beds=1, dims=(5.5, 3), scale=1, **kwargs):
        super().__init__(dims=dims, scale=scale, **kwargs)
        width, height = self.dims

        # create beds
        num_beds = max(1, min(num_beds, 2))
        self.beds = np.array([Bed(origin=[0, (height - 1) * b], scale=scale, rotation=90) for b in range(num_beds)])

        self.add_furniture(self.beds)
        self.bed_assignment = None

        self.furniture_origins = np.empty((len(self.furniture) - 1, 2))
        self.furniture_upper = np.empty((len(self.furniture) - 1, 2))
        self.get_furniture_grid()

    def assign_beds(self, ids):
        self.bed_assignment = ids[:len(self.beds)]

    def enter_world(self, n, idx=None, arriving_from=None):
        return [self.entry_point] * n

    def step(self, t):
        if not self.population:
            return

        # n = len(self.population)
        # if hasattr(self.population, "sleep"):
        #     # update staying duration
        #     stay_update = self.population.stay.ravel()
        #     if stay_update.any():
        #         self.population.accumulated_stay[stay_update] += 1
        #
        #     acc_stay = self.population.accumulated_stay.ravel()
        #     cur_stay = self.population.current_stay_duration.ravel()
        #     enough_staying = (acc_stay > cur_stay)
        #     enough_staying = (stay_update & enough_staying)
        #
        #     wake_up = (~self.population.sleep & self.population.in_bed).ravel()
        #     dressing = np.copy(wake_up)
        #     if wake_up.any():
        #         self.population.position[wake_up] = self.population.dress_pos[wake_up]
        #         self.population.target[wake_up] = self.population.dress_pos[wake_up]
        #         self.population.in_bed[wake_up] = False
        #         self.population.motion_mask[wake_up] = True
        #         self.population.busy[wake_up] = False
        #         # take time to dress
        #         self.population.stay[dressing] = True
        #         self.population.accumulated_stay[dressing] += 1
        #         # take 4 to 12 minutes
        #         stay_duration = partial(np.random.normal, global_time.make_time(minutes=8),
        #                                 global_time.make_time(minutes=2))((dressing.sum(), 1))
        #         self.population.current_stay_duration[dressing] = stay_duration
        #         self.population.stay[dressing] = True
        #     go = (wake_up & enough_staying)
        #     if go.any():
        #         self.population.stay[enough_staying] = False
        #         self.population.accumulated_stay[enough_staying] = 0
        #         self.population.current_stay_duration[enough_staying] = -np.inf
        #         self.population.target[go] = self.entry_point
        #         self.population.in_bed[go] = False
        #         self.population.motion_mask[go] = True
        #
        #     # Send people to sleep
        #     send_to_bed = (self.population.sleep & ~self.population.in_bed).ravel()
        #     at_dress_pos = np.zeros((n, 2), dtype=bool)
        #     at_dress_pos[send_to_bed] = self.population.position[send_to_bed] == self.population.dress_pos[send_to_bed]
        #     at_dress_pos = np.array([np.all(i) for i in at_dress_pos])
        #
        #     if send_to_bed.any():
        #         self.population.target[send_to_bed] = self.population.dress_pos[send_to_bed]
        #
        #     undress = (send_to_bed & at_dress_pos).ravel() & ~enough_staying
        #     if undress.any():
        #         # take 4 to 12 minutes
        #         stay_duration = partial(np.random.normal, global_time.make_time(minutes=8),
        #                                 global_time.make_time(minutes=2))((undress.sum(), 1))
        #         self.population.current_stay_duration[undress] = stay_duration
        #         self.population.stay[undress] = True
        #
        #     sleep = (send_to_bed & at_dress_pos).ravel() & enough_staying
        #     sleep = sleep & self.population.motion_mask.ravel()
            self.put_to_bed()
        #
        #     awake = (~self.population.sleep & ~self.population.in_bed & ~self.population.busy).ravel()
        #     if awake.any():
        #         self.population.target[awake] = np.nan
        #
        #     busy = (~self.population.sleep & ~self.population.in_bed & self.population.busy).ravel()
        #     if busy.any():
        #         self.population.target[busy] = self.entry_point

    def put_to_bed(self, ids):
        if ids.any():
            ids_in_room = self.bed_assignment.reshape(-1, 1) == ids
            beds = np.array([b.sleeping_pos + b.origin for b in self.beds])
            bed_assignment = ids_in_room.any(axis=1)
            ids = ids[ids_in_room.any(axis=0)]

            ids = (self.population.index.reshape(-1, 1) == ids).any(axis=1)
            self.population.motion_mask[ids] = False
            self.population.position[ids] = beds[bed_assignment]
            self.population.in_bed[ids] = True

    def get_out_of_bed(self, ids):
        if ids.any():
            ids = (self.population.index.reshape(-1, 1) == ids).any(axis=1)
            self.population.motion_mask[ids] = True
            self.population.in_bed[ids] = False
