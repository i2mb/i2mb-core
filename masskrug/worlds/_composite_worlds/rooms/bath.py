import numpy as np
from masskrug.worlds.furniture.furniture_bath.toilet import Toilet
from masskrug.worlds.furniture.furniture_bath.bathtub import Bathtub
from masskrug.worlds.furniture.furniture_bath.sink import Sink
from masskrug.worlds.furniture.furniture_bath.bathtub import Shower
from masskrug.utils import global_time
from functools import partial
from random import uniform
from ._room import BaseRoom

"""
    :param guest: Furniture in Bath, 0 = Shower, 1 = Bathtub, 2 = only toilet
    :type guest: int, optional
"""


class Bath(BaseRoom):
    def __init__(self, guest=0, dims=(3, 3), scale=1, **kwargs):
        super().__init__(dims=dims, scale=scale, **kwargs)
        width, height = self.dims[0], self.dims[1]
        offset = [0, 0, height, width, 0]
        rot = np.radians(self.rotation)

        self.occupied = False
        self.tasks = {}

        self.guest = guest
        if guest == 1:
            self.bathtub = Bathtub(self.rotation, scale=scale)
            # bathtub positioned in bottom left corner
            self.bathtub.origin = [offset[int(self.rotation / 90) + 1], offset[int(self.rotation / 90)]]

        elif guest == 0:
            self.bathtub = Shower(self.rotation, scale=scale)
            # shower positioned in bottom left corner
            self.bathtub.origin = [offset[int(self.rotation / 90) + 1], offset[int(self.rotation / 90)]]

        else:
            self.bathtub = None

        self.toilet = Toilet(self.rotation, scale=scale)
        # toilet positioned in bottom right corner
        toilet_offset = width - self.toilet.width * 1.5
        self.toilet.origin = [toilet_offset * np.cos(rot) + offset[int(self.rotation / 90) + 1],
                              toilet_offset * np.sin(rot) + offset[int(self.rotation / 90)]]

        self.sink = Sink(self.rotation, scale=scale)
        # sink positioned on right side of room
        sink_offset = width - self.sink.width
        self.sink.origin = [
            sink_offset * np.cos(rot) + offset[int(self.rotation / 90) + 1] - (height / 2) * np.sin(rot),
            sink_offset * np.sin(rot) + offset[int(self.rotation / 90)] + (height / 2) * np.cos(rot)]

        # rotate room
        if self.rotation == 90 or self.rotation == 270:
            self.dims[0], self.dims[1] = self.dims[1], self.dims[0]

        self.furniture += [self.sink, self.toilet, self.bathtub]

        self.furniture_origins = np.empty((len(self.furniture) - 1, 2))
        self.furniture_upper = np.empty((len(self.furniture) - 1, 2))
        self.get_furniture_grid()

        # target positions/areas for step method
        self.sink_area = self.get_sink_area()
        self.go_toilet_pos = self.toilet.sitting_pos + np.array(
            [-np.sin(rot) * self.toilet.length / 2, np.cos(rot) * self.toilet.length / 2])
        self.go_shower_pos = self.bathtub.showering_pos + np.array(
            [np.cos(rot) * self.bathtub.width / 2, -np.sin(rot) * self.bathtub.width / 2])

    def enter_world(self, n, idx=None, locations=None):
        self.occupied = True
        return [self.entry_point] * n

    def exit_world(self, idx):
        bool_ix = self.population.find_indexes(idx)

        self.population.motion_mask[bool_ix] = True
        self.population.stay[bool_ix] = False
        self.population.accumulated_stay[bool_ix] = 0
        self.population.current_stay_duration[bool_ix] = -np.inf
        for ix in self.population.index[bool_ix]:
            del self.tasks[ix]

    def get_sink_area(self):
        y0, y1 = self.furniture_origins[0][1], self.furniture_upper[0][1]
        x0, x1 = self.furniture_origins[0][0], self.furniture_upper[0][0]
        if self.rotation == 0:
            x1 = min(x0, x1)
            x0 = x1 - self.sink.width
        if self.rotation == 180:
            x0 = max(x0, x1)
            x1 = x0 + self.sink.width
        if self.rotation == 90:
            y1 = min(y0, y1)
            y0 = y1 - self.sink.length
        if self.rotation == 270:
            y0 = max(y0, y1)
            y1 = y0 + self.sink.width
        return [[min(x0, x1), max(x0, x1)], [min(y0, y1), max(y0, y1)]]

    def move_to_sink(self, n):
        # move agent somewhere in the sink area
        x = np.array([[uniform(self.sink_area[0][0], self.sink_area[0][1])] for i in range(n)])
        y = np.array([[uniform(self.sink_area[1][0], self.sink_area[1][1])] for i in range(n)])
        return np.concatenate((x, y), axis=1)

    def step(self, t):
        if not hasattr(self, "population"):
            return
        if not self.population:
            self.occupied = False
            return

        # update staying duration
        stay_update = self.population.stay.ravel()
        if stay_update.any():
            self.population.accumulated_stay[stay_update] += 1

        acc_stay = self.population.accumulated_stay.ravel()
        cur_stay = self.population.current_stay_duration.ravel()
        enough_staying = (acc_stay > cur_stay)
        enough_staying = stay_update & enough_staying

        # check if everyone has tasks
        keys = np.array([key for key in self.tasks.keys()])
        idxs = self.population.index
        entered = np.setdiff1d(idxs, keys)
        if len(entered) > 0:
            # zero is for going to toilet
            vals = np.zeros(len(entered))
            add_dict = dict(zip(entered, vals))
            self.tasks.update(add_dict)

        # send agents to their tasks
        values = np.array([self.tasks.get(idx) for idx in self.population.index])
        to_toilet = (values == 0) & ~enough_staying.ravel() & ~stay_update
        move_to_sink = ((values == 0) | (values == 2)) & enough_staying.ravel()
        short_sink = (values == 0) & enough_staying.ravel()
        long_sink = (values == 2) & enough_staying.ravel()
        if short_sink.any():
            self.occupied = False
        move_to_shower = (values == 1) & enough_staying.ravel()

        # just entered agents
        if to_toilet.any():
            self.population.target[to_toilet] = self.go_toilet_pos

        # move agents to their next task
        if enough_staying.any():
            self.population.stay[enough_staying] = False
            self.population.accumulated_stay[enough_staying] = 0
            self.population.current_stay_duration[enough_staying] = -np.inf
            self.population.motion_mask[enough_staying] = True
            if move_to_sink.any():
                self.population.target[move_to_sink] = self.move_to_sink(move_to_sink.sum())
                temp_dict = dict(zip(self.population.index[short_sink], np.full(short_sink.sum(), 1)))
                self.tasks.update(temp_dict)
                temp_dict = dict(zip(self.population.index[long_sink], np.full(long_sink.sum(), 3)))
                self.tasks.update(temp_dict)
            if self.bathtub is not None:
                self.population.target[move_to_shower] = self.go_shower_pos
                temp_dict = dict(zip(self.population.index[move_to_shower], np.full(move_to_shower.sum(), 2)))
                self.tasks.update(temp_dict)

        at_target = np.isclose(self.population.target, self.population.position)
        at_target = np.array([all(i) for i in at_target])
        at_target = at_target & self.population.motion_mask.ravel()
        leaving = self.population.target == self.entry_point
        leaving = np.array([all(i) for i in leaving])
        task = at_target & ~leaving

        if task.any():
            values = np.array([self.tasks.get(idx) for idx in self.population.index])
            # go to toilet
            toilet = (values == 0) & task
            if toilet.any():
                self.population.motion_mask[toilet] = False
                self.population.target[toilet] = self.toilet.sitting_pos
                # taking one to five minutes
                stay_duration = partial(np.random.normal, global_time.make_time(minutes=3),
                                        global_time.make_time(minutes=1))((toilet.sum(), 1))
                self.population.current_stay_duration[toilet] = stay_duration
            # go to sink
            short_sink = (values == 1) & task
            if short_sink.any():
                # taking two to six minutes
                stay_duration = partial(np.random.normal, global_time.make_time(minutes=4),
                                        global_time.make_time(minutes=1))((short_sink.sum(), 1))
                self.population.current_stay_duration[short_sink] = stay_duration
            long_sink = (values == 3) & task
            if long_sink.any():
                # stay till leaving
                self.population.current_stay_duration[long_sink] = np.inf
            # go showering
            shower = (values == 2) & task
            if shower.any():
                self.population.motion_mask[shower] = False
                self.population.target[shower] = self.bathtub.showering_pos
                # taking four to twelve minutes
                stay_duration = partial(np.random.normal, global_time.make_time(minutes=8),
                                        global_time.make_time(minutes=2))((shower.sum(), 1))
                self.population.current_stay_duration[shower] = stay_duration
            self.population.stay[task] = True

        return
