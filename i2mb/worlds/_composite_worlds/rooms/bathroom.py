from functools import partial

import numpy as np

import i2mb.activities.activity_descriptors
from i2mb.activities.base_activity import ActivityNone
from i2mb.utils import global_time
from i2mb.utils.distributions import TemporalLinkedDistribution
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

        shower_on_distribution = partial(np.random.randint, 1, global_time.make_time(minutes=15))
        shower_tld = TemporalLinkedDistribution(shower_on_distribution, global_time.make_time(hour=6))


        self.activities = [
            i2mb.activities.activity_descriptors.Toilet(location=self, device=self.toilet,
                                                        duration=global_time.make_time(minutes=5),
                                                        blocks_location=True,
                                                        blocks_for=global_time.make_time(hour=1)
                                                        ),
            i2mb.activities.activity_descriptors.Sink(location=self, device=self.sink,
                                                      duration=global_time.make_time(minutes=5),
                                                      blocks_for=global_time.make_time(minutes=45)),

            i2mb.activities.activity_descriptors.Shower(location=self, device=self.bathtub,
                                                        duration=shower_tld.sample_on,
                                                        blocks_for=shower_tld.sample_off,
                                                        blocks_location=True,
                                                        )]

        self.local_activities.extend(self.activities)
        self.descriptor_idx = {}

    def post_init(self, base_file_name=None):
        self.descriptor_idx = {d.activity_class.id: d for d in self.activities}

    def start_activity(self, idx, activity_id):
        if activity_id not in self.descriptor_idx:
            return

        bool_ix = self.population.find_indexes(idx)
        descriptor = self.descriptor_idx[activity_id]
        device = descriptor.device
        self.population.position[bool_ix] = device.get_activity_position()
