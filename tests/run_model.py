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

from collections import deque
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from i2mb.engine.core import Engine
from i2mb.interactions.digital_contact_tracing import ContactTracing
from i2mb.interventions.contact_isolation import ContactIsolationIntervention
from i2mb import NullModel as NM
from i2mb.engine.agents import AgentList
from i2mb.motion.random_motion import RandomMotion
from i2mb.pathogen.base_pathogen import UserStates
from i2mb.pathogen.infection import CoronaVirus
from i2mb.utils import global_time
from i2mb.worlds import SquareWorld, Home
from i2mb.worlds.g_pylons import GravityPylons

CM = rcParams["axes.prop_cycle"].by_key()['color']
LEGEND_LABELS = [u.name for u in UserStates]
LEGEND_LABELS = ["Susceptibles",
                 "Inmunes",
                 "Fallecidos",
                 "En Incubación",
                 "Sin síntomas",
                 "Infecciosos"]


def legend():
    custom_scat = [Line2D([], [], color='gray') for l in LEGEND_LABELS]
    for i, r in enumerate(LEGEND_LABELS):
        l1 = custom_scat[i]
        l1.set_marker('o')
        l1.set_markersize(5)
        # l1.set_markevery(-10)
        l1.set_markeredgecolor(CM[i])
        l1.set_alpha(0.7)
        l1.set_markerfacecolor(CM[i])

    ax.legend(custom_scat, LEGEND_LABELS,
              bbox_to_anchor=(1.0, 1),
              loc='upper left'
              )


def init():
    ax.set_xlim(-5, world.dims[0] + 5)
    ax.set_ylim(-5, world.dims[1] + 5)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), *WORLD_BOX, fill=False, linewidth=1.2))
    # w = world.containment[1]
    # ax.add_patch(Rectangle(world.containment[:2], w, w, fill=False, linewidth=1.2))

    ax1.set_xlim(0, NUM_FRAMES)
    ax1.set_ylim(0, NUM_PARTICLES)

    density = 5 * TIME_SCALAR
    for x in range(1, int(NUM_FRAMES / density) + 1):
        ax1.axvline(x * density, color="k", alpha=0.5)

    ax2.set_xlim(0, NUM_FRAMES)
    ax2.set_ylim(0, 2)
    ax2.set_xlabel("Tiempo")
    ax2.legend(["R", "Rrt"])
    bb = ax.get_position()
    # bb.y0 -= 0.2
    bb.x0 -= 0.025
    bb.x1 = 0.75
    ax.set_position(bb)
    ax.set_axis_off()

    bb = ax1.get_position()
    bb.x0 -= 0.25
    bb.x1 += 0.25

    for l in lines:
        l.set_data(deque([0]), deque([0]))

    legend()
    return scat, r_text


def update(frame):
    next(pos_gen)
    ct = cv = cti = gp = positions = h = None

    positions = population.position

    state = sl = pi = ni = None
    if cv is not None:
        state, sl, pi, ni = cv
    scat.set_offsets(positions)
    # if frame == 5:
    #     covid.introduce_pathogen(1, frame)

    r_c = covid.r_current()
    day = global_time.days(frame)
    hour = global_time.hour(frame)
    minute = global_time.minute(frame)
    r_text.set_text(f"{day}D - {hour}:{minute:02d}")
    points = covid.get_totals()
    order = [UserStates.infected, UserStates.asymptomatic, UserStates.incubation, UserStates.susceptible,
             UserStates.immune, UserStates.deceased]
    for d, t in zip(data, order):
        d.append(points[t])

    time.append(frame)
    ax1.collections.clear()
    ax1.stackplot(time, *data, colors=[CM[u] for u in order])

    new_contacts = {}
    if ct is not None:
        new_contacts = {id_: set(cl[0].contacts).difference(unique_contacts_history.get(id_, set()))
                        for id_, cl in enumerate(ct)}
        contacts_history.append({id_: len(v) for id_, v in new_contacts.items() if len(v) > 0})

    for id_, v in new_contacts.items():
        unique_contacts_history.setdefault(id_, set()).update(v)

    if state is not None:
        scat.set_color([CM[s] for s in state.ravel().astype(int)])

    if ni is not None:
        cni = r_data[2][-1] + ni

    for ix, d in enumerate([covid.r(), covid.r_current(), 0]):
        r_data[ix].append(d)
        r_lines[ix].set_data(time, r_data[ix])

    artists = list(lines) + list(r_lines) + [scat, r_text]
    ax2.set_ylim(0, max(r_data[0]) * 1.1)

    # if frame % 100 == 0:
    #     print(covid.waves)
    #     print(ct_intervention.num_isolations.ravel())
    #     print(ct_intervention.isolated_fp.ravel())

    return artists


if __name__ == "__main__":
    TIME_SCALAR = 8 * 24
    NUM_FRAMES = int(180 * TIME_SCALAR)
    NUM_PARTICLES = 4
    LENGTH = 6
    HEIGHT = 8
    WORLD_BOX = (LENGTH, HEIGHT)

    BEACONS = np.array([[LENGTH / 6, HEIGHT / 6],
                        [LENGTH / 6, HEIGHT / 6 * 5],
                        [LENGTH / 2, HEIGHT / 2],
                        [LENGTH / 6 * 5, HEIGHT / 6],
                        [LENGTH / 6 * 5, HEIGHT / 6 * 5]])

    global_time.ticks_hour = 8

    fig, axs = plt.subplots(3, 1)
    ax: plt.Axes
    ax, ax1, ax2 = axs

    # pos_gen = random_walk(1., 50, (100, 50))

    # Data fro plots
    data = [deque([0]) for u in UserStates]
    r_data = [deque([0]), deque([0]), deque([0])]
    time = deque([0])
    contacts_history = []
    unique_contacts_history = {}

    # Setup engine models
    population = AgentList(NUM_PARTICLES)

    # world = SquareWorld(WORLD_BOX, population)
    world = Home(dims=(6, 8), population=population, always_on=False, gain=10)
    motion = RandomMotion(world, population, step_size=0.2, )
    g_pylons = GravityPylons(beacons=BEACONS, population=population, world=world,
                             radius=1, gain=10)

    # Contact tracing model
    c_tracing = ContactTracing(radius=5, population=population, duration=2,
                               track_time=30 * TIME_SCALAR,
                               coverage=1,
                               false_positives=0,
                               false_negatives=0
                               )

    # Deceased duration distribution
    dd = partial(np.random.normal, 14 * TIME_SCALAR, 7 * TIME_SCALAR)
    id = partial(np.random.normal, 12 * TIME_SCALAR, 2 * TIME_SCALAR)

    # Deceased model
    covid = CoronaVirus(radius=2, exposure_time=2, population=population,
                        asymptomatic_p=0.4,
                        death_rate=(0.05, 0.05),
                        duration_distribution=dd,
                        incubation_distribution=id,
                        icu_beds=10)

    ct_intervention = ContactIsolationIntervention(delay=3 * TIME_SCALAR, population=population, world=world,
                                                   quarantine_duration=14 * TIME_SCALAR,
                                                   test_to_exit=True, test_duration=TIME_SCALAR)

    # engine = Engine([c_tracing, covid, ct_intervention, NM, motion], debug=True)
    engine = Engine([motion, world], debug=True)
    # engine = Engine([c_tracing, covid, NM, g_pylons, motion])
    # engine = Engine([c_tracing, covid, NM, NM, motion])

    # Init plots
    pos_gen = engine.step()
    next(pos_gen)
    positions = population.position

    scat = ax.scatter(*positions.T, s=16)
    ax.scatter(*g_pylons.beacons.T, marker="+", color="b")
    lines = ax1.plot([0], [[0 for u in UserStates]])
    r_lines = ax2.plot([0], [[0, 0, 0]])
    r_text = ax.text(1.02, 0.02, "R", transform=ax.transAxes)
    contacts_list = []
    ani = FuncAnimation(ax.get_figure(), update, frames=NUM_FRAMES, interval=1, repeat=False,
                        init_func=init, blit=False)

    plt.show(block=True)

    engine.finalize()

    df = pd.DataFrame(engine.debug_timer)
    df.mean().plot(kind='barh')
    print(df.head())
    print(df.describe())
    plt.show(block=True)

    # infected = sum((population.state == UserStates.infected) |
    #                (population.state == UserStates.asymptomatic))
    # immune = sum(population.state == UserStates.immune)
    # dead = sum((population.state == UserStates.deceased))
    # print(f"Infected: {infected}")
    # print(f"Susceptible: {sum((population.state == UserStates.susceptible))}")
    # print(f"Immune: {sum((population.state == UserStates.immune))}")
    # print(f"Deceased: {dead}")
    # print(f"Death rate: {dead / (infected + immune)}")
    # print(f"R: {covid.r()}")
    # print(f"Totals: {covid.get_totals()}")
    # print(f"Average contacts: {np.mean([np.mean(list(v.values())) for v in contacts_history if len(v.values()) > 0])}")
