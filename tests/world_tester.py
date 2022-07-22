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
from i2mb.engine.relocator import Relocator
from i2mb.motion.random_motion import RandomMotion
from i2mb.utils import global_time, time
from i2mb.worlds import CompositeWorld, Apartment, BaseRoom, LivingRoom, Restaurant, Corridor
from i2mb.worlds._area import Area
from i2mb.worlds.office import Office

global_time.ticks_scalar = 60 / 5 * 24


class WorldBuilder:
    def __init__(self, world_cls=CompositeWorld, world_kwargs=None, rotation=0, sim_duration=1000,
                 update_callback=None, no_gui=False, no_animation=False, population=None, use_office=False):
        if update_callback is None:
            update_callback = self.__update_callback

        Area.reset_id_map()
        self.update_callback = update_callback
        self.worlds = []
        self.fig = None
        self.ax = None
        self.ani = None
        self.sim_duration = sim_duration
        self.animation_finished = False
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

        self.relocator = Relocator(self.population, self.universe)

        # Engine
        motion = RandomMotion(self.population, step_size=0.2)
        self.engine = Engine([motion] + self.worlds, debug=True)
        global_time.set_sim_time(0)

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

            self.relocator.move_agents(self.population.index[start:end], w)

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
        return self.ax.scatter(*self.relocator.get_absolute_positions().T, **kwargs),

    def process_stop_criteria(self, frame):
        return frame >= self.sim_duration

    def frame_generator(self):
        for frame, _ in enumerate(self.engine.step()):
            # Stopping criteria
            stop = self.process_stop_criteria(frame)
            if stop:
                self.animation_finished = True
                return

            yield frame

    def update(self, frame):
        self.update_callback(self, frame)
        return self.draw_population()


class WorldBuilderTestGui(TestCase):
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
            WorldBuilder(world_cls=Office, world_kwargs=dict(dims=(10, 10)), rotation=rot)

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


class WorldBuilderTestsNoGui(TestCase):
    def test_ids(self):
        w = WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), no_gui=True)
        from i2mb.worlds import World
        self.assertEqual(len(Area.list_all_areas(w.universe)),
                         len([area for area in w.universe._Area__id_map.values()]))

    def test_index_creation(self):
        w = WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), no_gui=True)
        self.assertTrue((w.universe.region_index[0, :] == [-1, 0, -1]).all(),
                        msg=f"First index position should be [-1, 0, -1], got {w.universe.region_index[0, :]} instead")

        self.assertEqual(w.universe.parent, None, msg="Universe falsh parent")
        self.assertEqual(w.universe.index, len(w.universe.region_index) - 1, msg="Universe false index")

        for region in w.universe.list_all_regions():
            selection = w.universe.region_index[:, 0] == region.id
            self.assertEqual(selection.sum(), 1, msg=f"Region {region.id, region} is registered more "
                                                     f"than once in the index")
            index_entry = w.universe.region_index[selection, :].ravel()
            expected = [region.id, region.parent, region]
            if region.parent is None:
                expected[1] = 0
            else:
                expected[1] = region.parent.index

            self.assertListEqual(list(index_entry), expected)

    def test_global_time_update(self):
        w = WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), no_gui=True)
        t = time()
        for i, _ in enumerate(w.engine.step()):
            if i == 10:
                break

        t2 = time()
        self.assertTrue(t == 0, msg="Global Simulation Time is not starting at 0")
        self.assertEqual(10, t2, msg="Global Simulation Time is at 10 after 10 steps")

    def test_apartment_entry(self):
        w = WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), no_gui=True)
        pop1, pop2, pop3 = [ap.get_entrance_sub_region().population for ap in w.worlds]
        assert w.population.at_home.all()
        assert (pop1.index == w.population.index[:5]).all()
        assert (pop2.index == w.population.index[5:]).all()

        baseline = {Apartment: np.ones(len(w.population)), Corridor: np.ones(len(w.population))}
        self.assertDictWithArrays(baseline, w.relocator.visit_counter)

    def assertDictWithArrays(self, baseline, dict_):
        self.assertSequenceEqual(baseline.keys(), dict_.keys())
        for k, v in baseline.items():
            self.assertListEqual(list(v), list(dict_[k]))

    def test_apartment_exit(self):
        w = WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), no_gui=True)
        pop1, pop2, pop3 = [ap.population for ap in w.worlds]

        idx = pop1.index[-1]
        w.relocator.move_agents(pop1.index[-1:], w.worlds[2])
        assert (w.worlds[0].population.index == w.population.index[:4]).all()
        self.assertTrue(~w.population.at_home.all())

        baseline = {Apartment: np.ones(len(w.population)),
                    Corridor: np.ones(len(w.population)),
                    type(w.worlds[2]): np.zeros(len(w.population))}
        baseline[type(w.worlds[2])][idx] += 1
        self.assertDictWithArrays(baseline, w.relocator.visit_counter)

    def test_unified_index(self):
        w = WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), no_gui=True)

        # Test Block Propagation
        ids = [3, 5, 6]
        regions_idx = np.unique([r.index for r in w.population.location[ids]])
        w.universe.block_locations(regions_idx, True)

        self.assertTrue(w.population.location[ids][0].blocked_locations[1])
        self.assertTrue(w.population.location[ids][0].blocked)

        # test accessing via children
        w.population.location[ids][0].blocked = False
        w.population.location[ids][1].blocked = False
        self.assertFalse(w.universe.blocked_locations.any())

    def test_movement_cancellation_due_to_locked_regions(self):
        w = WorldBuilder(world_cls=Apartment, world_kwargs=dict(num_residents=6), no_gui=True)

        # Test Block Propagation
        ids = [5, 6]

        # Block Appartment 0
        w.universe.regions[0].blocked = True

        # Move ID 5 from apt 1 to apt 0 region 3 move should eb cancelled.
        destination = w.universe.regions[0].regions[3]
        origin = w.relocator.location[ids].tolist()
        w.relocator.move_agents(ids, destination)

        self.assertListEqual(origin, w.relocator.location[ids].tolist())

        # Move ID 3 from Corridor to destination, moved should be allowed
        w.relocator.move_agents([3], destination)
        self.assertTrue((w.relocator.location[[3]] == destination).all())

        # Test move is possible after unlock
        w.universe.regions[0].blocked = False
        w.relocator.move_agents(ids, destination)
        self.assertTrue((w.relocator.location[[5, 6]] == destination).all())

    def test_changing_rooms(self):
        return

