#  dct_mct_analysis
#  Copyright (C) 2021  FAU - RKI
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from argparse import ArgumentParser
from importlib import import_module
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from masskrug.engine.core import Engine

from masskrug.engine.particle import ParticleList
from masskrug.motion.random_motion import RandomMotion
from masskrug.utils import global_time
from masskrug.worlds import CompositeWorld, Apartment, BaseRoom, LivingRoom
from matplotlib.animation import FuncAnimation

global_time.ticks_scalar = 60 / 5* 24


class WorldBuilder:
    def __init__(self, world_cls=CompositeWorld, world_kwargs=None, rotation=0, sim_duration=1000, update_callback=None):
        if update_callback is None:
            update_callback = self.__update_callback

        self.update_callback = update_callback
        self.worlds = []
        self.fig = None
        self.ax = None
        self.ani = None
        self.sim_duration = sim_duration
        if world_kwargs is None:
            world_kwargs = {}

        for w in range(2):
            world_kwargs["origin"] = (rotation in [90, 270] and (1 + (1 + 7) * w, 1) or (1, 1 + (1 + 7) * w))
            world_kwargs["rotation"] = rotation
            self.worlds.append(world_cls(**world_kwargs))

        self.population = ParticleList(10)
        self.universe = CompositeWorld(population=self.population, regions=self.worlds, origin=[0, 0])
        self.universe.dims += 1

        # Engine
        motion = RandomMotion(self.universe, self.population, step_size=0.2)
        self.engine = Engine([motion] + self.worlds, debug=True)

        self.assign_agents_to_worlds()
        self.create_animation_engine()

    @staticmethod
    def __update_callback(self, frame):
        return

    def assign_agents_to_worlds(self):
        self.universe.move_agents(self.population.index[5:], self.worlds[1])
        start = 0
        end = 5
        for w in self.worlds:
            self.universe.move_agents(self.population.index[start:end], w)
            if hasattr(w, "move_home"):
                w.move_home(self.population[start:end])

            start += 5
            end += 5

    def create_animation_engine(self):
        self.fig, self.ax = plt.subplots(1)

        self.ax.set_aspect(1)
        self.ax.axis("off")
        self.universe.draw_world(self.ax, bbox=False)

        # Mark Point of entry
        self.draw_population(marker="x", s=20, color="red")

        w, h = self.universe.dims
        self.ax.scatter(*np.array([[0, 0], [w, h], [-w, h], [-w, -h], [w, -h],
                                   [h, w], [-h, w], [-h, -w], [h, -w]]).T, s=16)
        self.fig.tight_layout()
        self.ani = FuncAnimation(self.fig, self.update, frames=self.frame_generator,
                                 # fargs=(,),
                                 # init_func=init,
                                 interval=0.5,
                                 repeat=False,
                                 blit=True
                                 )

    def draw_population(self, **kwargs):
        kwargs.setdefault("color", "b")
        kwargs.setdefault("s", 10)
        return self.ax.scatter(*self.universe.get_absolute_positions().T, **kwargs),

    def process_stop_criteria(self, frame):
        return frame >= self.sim_duration

    def frame_generator(self):
        for frame, _ in enumerate(self.engine.step()):
            # Stopping criteria
            stop = self.process_stop_criteria(frame)
            if stop:
                return

            yield frame

    def update(self, frame):
        self.update_callback(self, frame)
        return self.draw_population()


class WorldBuilderTest(TestCase):
    def test_apartment_rotation_entrance_rand_motion(self):
        for rot in [0, 90, 180, 270]:
            WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), rotation=rot)

        plt.show(block=True)

    def test_base_room_rotation_entrance_rand_motion(self):
        for rot in [0, 90, 180, 270]:
            WorldBuilder(world_cls=BaseRoom, world_kwargs=dict(dims=(4, 7)), rotation=rot)

        plt.show(block=True)

    def test_livingroom_rotation_entrance_rand_motion(self):
        for rot in [0, 90, 180, 270]:
            WorldBuilder(world_cls=LivingRoom, world_kwargs=dict(num_seats=6), rotation=rot)

        plt.show(block=True)
