#  i2mb-core
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
from enum import IntEnum


class ActivityProperties(IntEnum):
    """Returns the index of the property"""
    start = 0
    duration = 1
    elapsed = 2
    accumulated = 3
    in_progress = 4
    blocked_for = 5
    location = 6


class ActivityDescriptorProperties(IntEnum):
    """Returns the index of the ActivityDescriptor properties"""
    act_idx = 0
    start = 1
    duration = 2
    priority_level = 3
    block_for = 4
    location_ix = 5
    blocks_location = 6
    blocks_parent_location = 7
    interruptable = 8
    descriptor_id = 9


class TypesOfLocationBlocking(IntEnum):
    """We consider three ways to block locations. The `shared` mode permits other agents in the location when
    blocking. In contrast, the `rejecting` mode will expel all agents from the current location. In `wait` mode,
    an agent will wait for the location to become available. """
    no_blocking = 0
    shared = 1
    wait = 2
    rejecting = 3


