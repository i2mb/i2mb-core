from argparse import ArgumentParser
from importlib import import_module

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from i2mb.worlds import CompositeWorld

import matplotlib.pyplot as plt
import numpy as np

from i2mb.engine.agents import AgentList

if __name__ == "__main__":
    parser = ArgumentParser(description='Helps design new worlds.')
    parser.add_argument("--world", help="Specify the world's class name.")
    module_name = "i2mb.worlds"

    args = parser.parse_args()
    world_name = args.world
    # world_name = "Apartment"

    # BAR / Restaurant

    mod = import_module(module_name)
    World = getattr(mod, world_name)  # type: CompositeWorld()
    population = AgentList(10)
    for rot in [0,
                90, 180, 270
                ]:
        world = World(
            # dims=(5.5 , 4),
            origin=(0, 0),
            dims=(20, 20)
        )
        world.rotate(rot)

        # world.origin = [1, 1]

        # world.move_agents(population.index, world)
        # world.sit_agents(population.index[:2])
        # world.sit_agents(population.index[2:3])
        # world.sit_agents(population.index[3:10])
        # world.sit_agents(population.index[10:12])
        # world.sit_agents(population.index[12:])

        fig, ax = plt.subplots(1)

        ax.set_aspect(1)
        ax.axis("off")
        # ax.set_xlim(0, 20 * 1.05)
        # ax.set_ylim(0, 20 * 1.05)
        print("main Loop", world.dims)
        world.draw_world(ax, bbox=False)
        w, h = world.dims
        ax.set_xlim(0, w * 1.05)
        ax.set_ylim(0, h * 1.05)

        # ax.scatter(*np.array([[0, 0], [w, h], [-w, h], [-w, -h], [w, -h],
        #                       [h, w], [-h, w], [-h, -w], [h, -w]]).T, s=16)
        # fig.tight_layout()

    plt.show(block=True)
