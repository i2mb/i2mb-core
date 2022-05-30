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

from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from i2mb.engine.agents import AgentList
from i2mb.engine.core import Engine
from i2mb.motion.random_motion import RandomMotion
from i2mb.utils import global_time
from i2mb.worlds import CompositeWorld, Apartment, BaseRoom, LivingRoom, Restaurant, Corridor
from i2mb.worlds._area import Area
from i2mb.worlds.office import Office

global_time.ticks_scalar = 60 / 5 * 24


class WorldBuilder:
    def __init__(self, world_cls=CompositeWorld, world_kwargs=None, rotation=0, sim_duration=1000,
                 update_callback=None, no_gui=False, no_animation=False, population=None, use_office=False):
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

        world_kwargs = {"origin": (rotation in [90, 270] and (1 + (1 + 7) * (w + 1), 1) or (1, 1 + (1 + 7) * (w + 1))),
                        "rotation": rotation, "dims": (4, 4)}
        if use_office:
            world_kwargs["dims"] = (10, 10)
            self.worlds.append(Office(**world_kwargs))
        else:
            self.worlds.append(CompositeWorld(**world_kwargs))

        if population is None:
            population = AgentList(10)

        self.population = population
        self.universe = CompositeWorld(population=self.population, regions=self.worlds, origin=[0, 0])
        self.universe.dims += 1

        # Engine
        motion = RandomMotion(self.universe, self.population, step_size=0.2)
        self.engine = Engine([motion] + self.worlds, debug=True)

        self.assign_agents_to_worlds()
        if not no_gui:
            self.create_animation_engine(no_animation)

    @staticmethod
    def __update_callback(self, frame):
        return

    def assign_agents_to_worlds(self):
        start = 0
        end = 5
        for w in self.worlds:
            if hasattr(w, "move_home"):
                w.move_home(self.population[start:end])
                self.population.home[start:end] = w

            if hasattr(w, "assign_beds"):
                w.assign_beds()

            self.universe.move_agents(self.population.index[start:end], w)

            start += 5
            end += 5

    def create_animation_engine(self, no_animation=False):
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
        if not no_animation:
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
    def test_ids(self):
        w = WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), no_gui=True)
        self.assertEqual(len(Area.list_all_regions(w.universe)), len(w.universe._Area__id_map))

    def test_apartment_entry(self):
        w = WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), no_gui=True)
        pop1, pop2, pop3 = [ap.get_entrance_sub_region().population for ap in w.worlds]
        assert w.population.at_home.all()
        assert (pop1.index == w.population.index[:5]).all()
        assert (pop2.index == w.population.index[5:]).all()

        baseline = {Apartment: np.ones(len(w.population)), Corridor: np.ones(len(w.population))}
        self.assertDictWithArrays(baseline, w.universe.visit_counter)

    def assertDictWithArrays(self, baseline, dict_):
        self.assertSequenceEqual(baseline.keys(), dict_.keys())
        for k, v in baseline.items():
            self.assertListEqual(list(v), list(dict_[k]))

    def test_apartment_exit(self):
        w = WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), no_gui=True)
        pop1, pop2, pop3 = [ap.population for ap in w.worlds]

        idx = pop1.index[-1]
        w.universe.move_agents(pop1.index[-1], w.worlds[2])
        assert (w.worlds[0].population.index == w.population.index[:4]).all()
        assert ~w.population.at_home.all()

        baseline = {Apartment: np.ones(len(w.population)),
                    Corridor: np.ones(len(w.population)),
                    type(w.worlds[2]): np.zeros(len(w.population))}
        baseline[type(w.worlds[2])][idx] += 1
        self.assertDictWithArrays(baseline, w.universe.visit_counter)

    def test_changing_rooms(self):
        return

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

    def test_office_build(self):
        for rot in [0, 90, 180, 270]:
            WorldBuilder(world_cls=Office, world_kwargs=dict(width=(10, 10)), rotation=rot)

        plt.show(block=True)

    def test_appartment_office_office_build(self):
        for rot in [0, 90, 180, 270]:
            WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), rotation=rot, use_office=True,
                         no_animation=True)

        plt.show(block=True)

    def test_office_restaurant(self):
        for rot in [0, 90, 180, 270]:
            WorldBuilder(world_cls=Restaurant, world_kwargs=dict(), rotation=rot)

        plt.show(block=True)
