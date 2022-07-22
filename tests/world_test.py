from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation

from i2mb.engine.core import Engine
from i2mb.engine.agents import AgentList
from i2mb.engine.relocator import Relocator
from i2mb.interactions.digital_contact_tracing import RegionContactTracing
from i2mb.interventions.contact_isolation import ContactIsolationIntervention
from i2mb.motion.random_motion import RandomMotion
from i2mb.pathogen.infection import RegionCoronaVirus
from i2mb.utils.visualization.world_view import draw_world
from i2mb.worlds.composite_world import CompositeWorld

CM = rcParams["axes.prop_cycle"].by_key()['color']


def update(frame):
    positions = next(pos_gen)

    state = population.state
    scat.set_offsets(world.get_absolute_positions())
    scat.set_color([CM[s] for s in state.ravel().astype(int)])

    if frame == 5:
        covid.introduce_pathogen(1, frame)

    return scat,


if __name__ == "__main__":
    NUM_PARTICLES = 300
    WORLD_BOX = (100, 100)
    TIME_SCALAR = 18

    # Setup engine models
    population = AgentList(NUM_PARTICLES)

    # world = SquareWorld(WORLD_BOX, population)
    l = WORLD_BOX[0] / 2
    isolation = CompositeWorld((l, l), origin=[WORLD_BOX[0] * 1.5, l / 2])
    roaming_area = CompositeWorld(WORLD_BOX)

    world = CompositeWorld(WORLD_BOX, population, regions=[roaming_area, isolation])

    relocator = Relocator(population, world)
    relocator.move_agents(slice(None), roaming_area)

    world.home_regions[:] = roaming_area
    world.containment_region[:] = isolation

    # world.move_agents(slice(50), roaming_area)
    motion = RandomMotion(world, population, step_size=0.5, )

    c_tracing = RegionContactTracing(radius=5, population=population, duration=2,
                                     track_time=30 * TIME_SCALAR,
                                     coverage=1,
                                     false_positives=0,
                                     false_negatives=0
                                     )

    # Deceased duration distribution
    dd = partial(np.random.normal, 14 * TIME_SCALAR, 7 * TIME_SCALAR)
    id = partial(np.random.normal, 12 * TIME_SCALAR, 2 * TIME_SCALAR)

    # Deceased model
    covid = RegionCoronaVirus(radius=2, exposure_time=2, population=population,
                              asymptomatic_p=0.4,
                              death_rate=(0.05, 0.05),
                              duration_distribution=dd,
                              incubation_distribution=id,
                              icu_beds=10)

    ct_intervention = ContactIsolationIntervention(delay=1 * TIME_SCALAR, population=population, world=world,
                                                   quarantine_duration=14 * TIME_SCALAR,
                                                   test_to_exit=True, test_duration=TIME_SCALAR)

    engine = Engine([c_tracing, covid, ct_intervention, motion], debug=True)

    # Init plots
    pos_gen = engine.step()

    fig, ax = plt.subplots(1, 1)
    scat = draw_world(world, ax)

    ani = FuncAnimation(ax.get_figure(), update, frames=500, interval=1, repeat=False,
                        blit=False)

    plt.show(block=True)

    df = pd.DataFrame(engine.debug_timer)
    df.mean().plot(kind='barh')
    print(df.head())
    print(df.describe())

    plt.show(block=True)
