from argparse import ArgumentParser
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np

from masskrug.engine.particle import ParticleList


if __name__ == "__main__":
    parser = ArgumentParser(description='Helps design new worlds.')
    parser.add_argument("--world", help="Specify the world's class name.")
    module_name = "masskrug.worlds"

    args = parser.parse_args()
    # world_name = args.world
    world_name = "Apartment"

    # BAR / Restaurant

    mod = import_module(module_name)
    World = getattr(mod, world_name)
    population = ParticleList(34)
    for rot in [0,
                90, 180, 270
                ]:
        world = World(
            # dims=(5.5 , 4),
            origin=(1, 2),
            num_residents=6,
            # sits=8,
            # num_beds=2,
            rotation=rot,
            # guest=1,
            # num_seats=8
        )
        # world.rotate_area(rot)

        # world.origin = [1, 1]

        # world.move_agents(population.index, world)
        # world.sit_particles(population.index[:2])
        # world.sit_particles(population.index[2:3])
        # world.sit_particles(population.index[3:10])
        # world.sit_particles(population.index[10:12])
        # world.sit_particles(population.index[12:])

        fig, ax = plt.subplots(1)

        ax.set_aspect(1)
        ax.axis("off")
        # ax.set_xlim(0, 20 * 1.05)
        # ax.set_ylim(0, 20 * 1.05)
        print("main Loop", world.dims)
        world.draw_world(ax, bbox=True)
        w, h = world.dims
        ax.scatter(*np.array([[0, 0], [w, h], [-w, h], [-w, -h], [w, -h],
                              [h, w], [-h, w], [-h, -w], [h, -w]]).T, s=16)
        fig.tight_layout()

    plt.show(block=True)
