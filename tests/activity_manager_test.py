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

from masskrug.utils import global_time
from masskrug.worlds import BedRoom
from tests.world_tester import WorldBuilder

global_time.ticks_scalar = 60 * 24


class TestActivityManager(TestCase):
    def test_in_bedroom(self):
        w = WorldBuilder(BedRoom, world_kwargs=dict(num_beds=2), sim_duration=global_time.make_time(day=4))
        activity_manager = LocalActivityManager(w.population)
