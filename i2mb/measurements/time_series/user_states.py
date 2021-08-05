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
import numpy as np
from i2mb.pathogen import UserStates


def get_awake_agents(population):
    return (~population.sleep & (population.state != UserStates.deceased)).ravel()


def get_location_names(population, filter_by):
    return np.array([type(loc).__name__.lower() for loc in population.location[filter_by]])


def get_location_contracted(population, locations):
    return (population.location_contracted == locations).sum(axis=0)


def get_number_of_isolated_agents(population):
    if not hasattr(population, "isolated"):
        return 0

    return population.isolated.sum()


def get_total_number_of_isolations(population):
    return population.num_isolations.sum()


def get_population_state_summary(population, user_states):
    return (population.state == user_states).sum(axis=0)


def get_false_positive_rate_of_isolated_agents(population):
    total_num_isolation = get_total_number_of_isolations(population)
    if total_num_isolation == 0:
        return 0

    else:
        return population.isolated_fp.sum() / total_num_isolation














