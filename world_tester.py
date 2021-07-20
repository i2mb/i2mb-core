from argparse import ArgumentParser
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
from masskrug.engine.core import Engine

from masskrug.engine.particle import ParticleList
from masskrug.motion.random_motion import RandomMotion
from masskrug.utils import global_time
from masskrug.worlds import CompositeWorld
from matplotlib.animation import FuncAnimation


def draw_population(ax_, world_, **kwargs):
    kwargs.setdefault("color", "b")
    kwargs.setdefault("s", 10)
    return ax_.scatter(*world_.get_absolute_positions().T, **kwargs)


def process_stop_criteria(frame):
    return frame >= 1000


def frame_generator(_engine):
    def _frame_gen():
        for frame, _ in enumerate(_engine.step()):
            # Stopping criteria
            stop = process_stop_criteria(frame)
            if stop:
                return

            yield frame

    return _frame_gen()


def update(ax_, universe_):
    def _update(frame):
        return draw_population(ax_, universe_),

    return _update


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
    for rot in [0, 90, 180, 270
                ]:
        worlds = [World(
            # dims=(5.5, 4),
            num_residents=6,
            origin=(rot in [90, 270] and (1 + (1 + 7) * w, 1) or (1, 1 + (1 + 7) * w)),
            # sits=8,
            # num_beds=2,
            rotation=rot,
            # guest=1,
            # num_seats=8
        ) for w in range(2)]

        population = ParticleList(10)
        universe = CompositeWorld(population=population,
                                  regions=worlds, origin=[0, 0])
        # universe.origin = 0, 0
        universe.dims += 1
        motion = RandomMotion(universe, population, step_size=0.2)
        global_time.ticks_scaler = 60 / 5 * 24
        engine = Engine([motion], debug=True)
        #
        universe.move_agents(population.index[:5], worlds[0])
        universe.move_agents(population.index[5:], worlds[1])

        fig, ax = plt.subplots(1)

        ax.set_aspect(1)
        ax.axis("off")
        universe.draw_world(ax, bbox=False)

        # Mark Point of entry
        draw_population(ax, universe, marker="x", s=20, color="red")

        w, h = universe.dims
        ax.scatter(*np.array([[0, 0], [w, h], [-w, h], [-w, -h], [w, -h],
                              [h, w], [-h, w], [-h, -w], [h, -w]]).T, s=16)
        fig.tight_layout()
        ani = FuncAnimation(fig, update(ax, universe), frames=frame_generator(engine),
                            # fargs=(,),
                            # init_func=init,
                            interval=0.5,
                            repeat=False,
                            blit=True
                            )

    plt.show()

