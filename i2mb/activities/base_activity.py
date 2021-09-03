#  i2mb
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
from itertools import cycle

import numpy as np


def activity_vectorized(func):
    def __wrapper(x):
        return np.vectorize(func, otypes=[ActivityPrimitive])(x)

    return __wrapper


class ActivityPrimitive:
    # Vectorized property names
    __keys = ["start", "duration", "elapsed", "accumulated", "in_progress"]

    def __init__(self, population):
        # Where the activity will take place
        self.population = population

        # Rank determines the activities that can interrupt other activities, higher rank interrupt lower rank.
        self.rank = 0

        n = len(self.population)
        self.location = np.full(n, None, dtype=object)

        # Device in location used during activity, e.g., bed
        self.device = np.full((n, 2), np.nan, dtype=float)

        # Link to Activity Descriptor
        self.descriptor = np.full(n, None, dtype=object)

        self.stationary = True

        # Precompute property index, and getter function
        for ix, prop_name in enumerate(ActivityPrimitive.__keys):
            setattr(self, f"{prop_name}_ix", ix)

        for ix, prop_name in enumerate(ActivityPrimitive.__keys):
            setattr(self, f"get_{prop_name}", self.__prop_getter__(prop_name))

        # Map properties into a contiguous 2D array
        self.__values = np.hstack([np.zeros((n, 1), dtype=int) for k in ActivityPrimitive.__keys])

    def __prop_getter__(self, prop_name):
        ix = getattr(self, f"{prop_name}_ix")

        def get_me():
            return self.__values[:, ix]

        return get_me

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, value):
        self.__values = value

    @property
    def keys_index(self):
        return enumerate(ActivityPrimitive.__keys)

    @classmethod
    def keys(cls):
        return cls.__keys

    def start_activity(self, t, ranks):
        start_activity = (self.get_duration() > 0) & (self.get_start() <= t)
        start_activity &= ranks <= self.rank
        start_activity &= self.population.at_home
        location_available = self.location != None
        mask = start_activity & location_available
        self.get_in_progress()[start_activity] = True

        return self.population.index[mask], self.location[mask]

    def finalize_start(self, ids):
        self.population.position[ids] = self.device[ids, :]
        if self.stationary:
            self.population.motion_mask[ids] = False

    def stop_activity(self, t):
        stop_activity = (self.get_elapsed() > self.get_duration()).ravel()
        self.get_in_progress()[stop_activity] = False
        self.get_elapsed()[stop_activity] = 0
        self.get_duration()[stop_activity] = 0
        self.location[stop_activity] = None
        self.device[stop_activity, :] = np.nan
        if self.stationary:
            self.population.motion_mask[stop_activity] = True

        mask = stop_activity & (self.descriptor != None)
        for act_desc in self.descriptor[mask]:
            act_desc.available = True

        self.descriptor[mask] = None

        return self.population.index[stop_activity]

    def __repr__(self):
        return f"Activity  {type(self)})"


class ActivityNone(ActivityPrimitive):
    def __init__(self, population):
        super().__init__(population)
        self.stationary = False


class ActivityDescriptor:
    """Agents can perform activities while in a location. The ActivityDescriptor class is a base class which describes
    where in the current location an activity can be performed."""
    def __init__(self, location=None, device=None, pos_id=None):
        # Where the activity will take place
        self.location = location

        # Device in location used during activity, e.g., bed
        self.device = device

        # Activity availability
        self.available = True

        # Keep track of how much this particular activity was used.
        self.accumulated = 0

        # Keep track of how much this particular activity was used.
        self.activity_class = ActivityNone
        #
        if pos_id is None:
            pos_id = 0
        self.pos_id = pos_id

    def __repr__(self):
        return f"Activity  {type(self)}:(location:{self.location}, device:{self.device}, "\
                 f"accumulated:{self.accumulated})"


class CompoundActivityDescriptor:
    """Describes a sequence of activity descriptors. """
    def __init__(self, activity_descriptors):
        self.activity_descriptors = activity_descriptors
        self.__activity_cycle = cycle(self.activity_descriptors)

    def __next__(self):
        return next(self.__activity_cycle)


class ActivityList:
    def __init__(self, population):
        population_size = len(population)
        self.activity_values = np.empty((population_size, len(ActivityPrimitive.keys()), 1),  dtype=object)
        self.activities = [ActivityNone(population)]
        self.activity_types = [ActivityNone]

        # Register vectorized property index getters
        for ix, prop_name in enumerate(ActivityPrimitive.keys()):
            setattr(self, f"{prop_name}_ix", ix)

        self.current_activity = np.zeros(len(population), dtype=int)

    # def __getitem__(self, item):
    #     return ActivityListView(item, self)

    def __repr__(self):
        return repr(self.activities)

    def __str__(self):
        return str(self.activities)

    def register(self, activity):
        self.activities.append(activity)
        self.activity_types.append(type(activity))
        self.activity_values = np.dstack([act.values for act in self.activities])
        activity.list = self

        # Connect activity values to the stack to free up memory and synchronize per activity operations with
        # ActivityList operations
        for ix, act in enumerate(self.activities):
            act.values = self.activity_values[:, :, ix]

    @property
    def shape(self):
        return self.activity_values.shape

    def get_current_activity_property(self, prop_ix):
        pop_size = self.activity_values.shape[0]
        index_vector = np.array(tuple(zip(range(pop_size), [prop_ix] * pop_size, self.current_activity)))
        index_vector = np.ravel_multi_index(index_vector.T, self.activity_values.shape)
        return self.activity_values.ravel()[index_vector]

    def set_activity_property(self, prop_ix, value, activity, ids=None):
        pop_size = len(activity)
        if ids is None:
            pop_size = self.activity_values.shape[0]
            ids = range(pop_size)

        index_vector = np.array(tuple(zip(ids, [prop_ix] * pop_size, activity)))
        index_vector = np.ravel_multi_index(index_vector.T, self.activity_values.shape)
        self.activity_values.ravel()[index_vector] = value

    def set_current_activity_property(self, prop_ix, value, ids=None):
        self.set_activity_property(prop_ix, value, self.current_activity, ids=None)


# class ActivityListView:
#     def __init__(self, range_, list_: ActivityList):
#         self.__list = list_
#         self.__range = range_
#
#     def __repr__(self):
#         return repr(self.activities)
#
#     def __str__(self):
#         return str(self.activities)
