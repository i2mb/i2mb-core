import numpy as np

import i2mb.activities.activity_descriptors
from i2mb.activities.base_activity import ActivityNone
from i2mb.utils import global_time
from i2mb.worlds.furniture.furniture_bath.bathtub import Bathtub
from i2mb.worlds.furniture.furniture_bath.bathtub import Shower
from i2mb.worlds.furniture.furniture_bath.sink import Sink
from i2mb.worlds.furniture.furniture_bath.toilet import Toilet
from ._room import BaseRoom

"""
    :param guest: Furniture in Bathroom, 0 = Shower, 1 = Bathtub, 2 = only toilet
    :type guest: int, optional
"""


class Bathroom(BaseRoom):
    def __init__(self, guest=0, dims=(3, 3), scale=1, **kwargs):
        super().__init__(dims=dims, scale=scale, **kwargs)

        self.occupied = False
        self.tasks = {}

        self.guest = guest
        if guest == 1:
            self.bathtub = Bathtub(rotation=90, origin=[0., 0.], scale=scale)

        elif guest == 0:
            self.bathtub = Shower(origin=[0., 0.], scale=scale)

        else:
            self.bathtub = None

        self.toilet = Toilet(rotation=180, origin=[0.2, self.height - 0.75])
        self.sink = Sink(rotation=90, origin=[(self.width - 1) / 2 + 0.7, self.height - 0.5], scale=scale)

        self.add_furniture([self.sink, self.toilet, self.bathtub])
        self.furniture_origins = np.empty((len(self.furniture) - 1, 2))
        self.furniture_upper = np.empty((len(self.furniture) - 1, 2))
        self.get_furniture_grid()
        self.activities = [
            i2mb.activities.activity_descriptors.Toilet(location=self, device=self.toilet,
                                                        duration=global_time.make_time(minutes=5),
                                                        blocks_location=True,
                                                        blocks_for=global_time.make_time(hour=3)
                                                        ),
            i2mb.activities.activity_descriptors.Sink(location=self, device=self.sink),
            i2mb.activities.activity_descriptors.Shower(location=self, device=self.bathtub,
                                                        duration=global_time.make_time(minutes=15),
                                                        blocks_for=global_time.make_time(hour=22),
                                                        blocks_location=True,
                                                        )]

        self.local_activities.extend(self.activities)
        self.descriptor_idx = {}

    def post_init(self, base_file_name=None):
        self.descriptor_idx = {d.activity_id: d for d in self.activities}

    def start_activity(self, idx, activity_id):
        if activity_id == ActivityNone.id:
            return

        bool_ix = self.population.find_indexes(idx)
        descriptor = self.descriptor_idx[activity_id]
        device = descriptor.device
        self.population.position[bool_ix] = device.get_activity_position()

    def step(self, t):
        return

        # # update staying duration
        # stay_update = self.population.stay.ravel()
        # if stay_update.any():
        #     self.population.accumulated_stay[stay_update] += 1
        #
        # acc_stay = self.population.accumulated_stay.ravel()
        # cur_stay = self.population.current_stay_duration.ravel()
        # enough_staying = (acc_stay > cur_stay)
        # enough_staying = stay_update & enough_staying
        #
        # # check if everyone has tasks
        # keys = np.array([key for key in self.tasks.keys()])
        # idxs = self.population.index
        # entered = np.setdiff1d(idxs, keys)
        # if len(entered) > 0:
        #     # zero is for going to toilet
        #     vals = np.zeros(len(entered))
        #     add_dict = dict(zip(entered, vals))
        #     self.tasks.update(add_dict)
        #
        # # send agents to their tasks
        # values = np.array([self.tasks.get(idx) for idx in self.population.index])
        # to_toilet = (values == 0) & ~enough_staying.ravel() & ~stay_update
        # move_to_sink = ((values == 0) | (values == 2)) & enough_staying.ravel()
        # short_sink = (values == 0) & enough_staying.ravel()
        # long_sink = (values == 2) & enough_staying.ravel()
        # if short_sink.any():
        #     self.occupied = False
        # move_to_shower = (values == 1) & enough_staying.ravel()
        #
        # # just entered agents
        # if to_toilet.any():
        #     self.population.target[to_toilet] = self.go_toilet_pos
        #
        # # move agents to their next task
        # if enough_staying.any():
        #     self.population.stay[enough_staying] = False
        #     self.population.accumulated_stay[enough_staying] = 0
        #     self.population.current_stay_duration[enough_staying] = -np.inf
        #     self.population.motion_mask[enough_staying] = True
        #     if move_to_sink.any():
        #         self.population.target[move_to_sink] = self.move_to_sink(move_to_sink.sum())
        #         temp_dict = dict(zip(self.population.index[short_sink], np.full(short_sink.sum(), 1)))
        #         self.tasks.update(temp_dict)
        #         temp_dict = dict(zip(self.population.index[long_sink], np.full(long_sink.sum(), 3)))
        #         self.tasks.update(temp_dict)
        #     if self.bathtub is not None:
        #         self.population.target[move_to_shower] = self.go_shower_pos
        #         temp_dict = dict(zip(self.population.index[move_to_shower], np.full(move_to_shower.sum(), 2)))
        #         self.tasks.update(temp_dict)
        #
        # at_target = np.isclose(self.population.target, self.population.position)
        # at_target = np.array([all(i) for i in at_target])
        # at_target = at_target & self.population.motion_mask.ravel()
        # leaving = self.population.target == self.entry_point
        # leaving = np.array([all(i) for i in leaving])
        # task = at_target & ~leaving
        #
        # if task.any():
        #     values = np.array([self.tasks.get(idx) for idx in self.population.index])
        #     # go to toilet
        #     toilet = (values == 0) & task
        #     if toilet.any():
        #         self.population.motion_mask[toilet] = False
        #         self.population.target[toilet] = self.toilet.sitting_pos
        #         # taking one to five minutes
        #         stay_duration = partial(np.random.normal, global_time.make_time(minutes=3),
        #                                 global_time.make_time(minutes=1))((toilet.sum(), 1))
        #         self.population.current_stay_duration[toilet] = stay_duration
        #     # go to sink
        #     short_sink = (values == 1) & task
        #     if short_sink.any():
        #         # taking two to six minutes
        #         stay_duration = partial(np.random.normal, global_time.make_time(minutes=4),
        #                                 global_time.make_time(minutes=1))((short_sink.sum(), 1))
        #         self.population.current_stay_duration[short_sink] = stay_duration
        #     long_sink = (values == 3) & task
        #     if long_sink.any():
        #         # stay till leaving
        #         self.population.current_stay_duration[long_sink] = np.inf
        #     # go showering
        #     shower = (values == 2) & task
        #     if shower.any():
        #         self.population.motion_mask[shower] = False
        #         self.population.target[shower] = self.bathtub.showering_pos
        #         # taking four to twelve minutes
        #         stay_duration = partial(np.random.normal, global_time.make_time(minutes=8),
        #                                 global_time.make_time(minutes=2))((shower.sum(), 1))
        #         self.population.current_stay_duration[shower] = stay_duration
        #     self.population.stay[task] = True

        return
