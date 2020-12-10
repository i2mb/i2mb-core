from argparse import ArgumentParser
from importlib import import_module

import matplotlib.pyplot as plt

from masskrug.engine.particle import ParticleList

if __name__ == "__main__":
    parser = ArgumentParser(description='Helps design new worlds.')
    parser.add_argument("--world", help="Specify the world's class name.")
    module_name = "masskrug.worlds"

    args = parser.parse_args()
    world_name = args.world

    # BAR / Restaurant

    mod = import_module(module_name)
    World = getattr(mod, world_name)

    population = ParticleList(34)
    world = World(dims=(20, 20), origin=(10, 4), population=population)

    # world.move_particles(population.index, world)
    world.sit_particles(population.index[:2])
    world.sit_particles(population.index[2:3])
    world.sit_particles(population.index[3:10])
    world.sit_particles(population.index[10:12])
    world.sit_particles(population.index[12:])

    fig, ax = plt.subplots(1)
    fig.tight_layout()
    ax.set_aspect(1)
    ax.axis("off")
    ax.set_xlim(0, 20 * 1.05)
    ax.set_ylim(0, 20 * 1.05)

    world.draw_world(ax, bbox=True)
    ax.scatter(*world.get_absolute_positions().T, s=16)

    plt.show()
