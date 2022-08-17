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
from typing import Protocol, Any

import numpy as np

from i2mb.activities import ActivityProperties


def activity_vectorized(func):
    def __wrapper(x):
        return np.vectorize(func, otypes=[ActivityPrimitive])(x)

    return __wrapper


class ActivityPrimitive:
    id = -1

    def __init__(self, population):
        # Where the activity will take place
        self.population = population

        # Rank determines the activities that can interrupt other activities, higher rank interrupt lower rank.
        self.rank = 0

        # Activity id this value gets updated once the activity is registered with the list.
        self.id = -1

        n = len(self.population)
        # self.location = np.full(n, None, dtype=object)

        # Device in location used during activity, e.g., bed
        self.device = np.full((n, 2), np.nan, dtype=float)

        self.stationary = True

        # Map properties into a contiguous 2D array
        self.__values = np.hstack([np.zeros((n, 1), dtype=int) for _ in ActivityProperties])

        self.__stop_callback = []
        self.__start_callback = []

    def get_start(self) -> np.ndarray:
        return self.__values[:, ActivityProperties.start]

    def get_duration(self) -> np.ndarray:
        return self.__values[:, ActivityProperties.duration]

    def get_elapsed(self) -> np.ndarray:
        return self.__values[:, ActivityProperties.elapsed]

    def get_accumulated(self) -> np.ndarray:
        return self.__values[:, ActivityProperties.accumulated]

    def get_in_progress(self) -> np.ndarray:
        return self.__values[:, ActivityProperties.in_progress]

    def get_blocked_for(self) -> np.ndarray:
        return self.__values[:, ActivityProperties.blocked_for]

    def get_location(self) -> np.ndarray:
        return self.__values[:, ActivityProperties.location]

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, value):
        self.__values = value

    def start_activity(self, t, start_activity):
        start_activity = self.population.index[start_activity]
        self.run_start_callbacks(t, start_activity)

        if not hasattr(self.population, "location"):
            return

        locations = set(self.population.location[start_activity])
        for loc in locations:
            idx = start_activity[self.population.location[start_activity] == loc]
            loc.start_activity(idx, self.id)
            self.get_location()[idx] = loc.index

    def finalize_start(self, ids):
        if not hasattr(self.population, "position"):
            return

        device_selector = ids[~np.isnan(self.device).any(axis=1)[ids]]
        if device_selector.any():
            self.population.position[device_selector] = self.device[device_selector, :]

        # Only change motion mask when made explicit.
        if self.stationary is True:
            self.population.motion_mask[ids] = False

    def stop_activity(self, t, stop_selector):
        # self.location[stop_selector] = None
        self.device[stop_selector, :] = np.nan
        if hasattr(self.population, "motion_mask"):
            self.population.motion_mask[stop_selector] = True

        if hasattr(self.population, "location"):
            locations = set(self.population.location[stop_selector])
            for loc in locations:
                loc.stop_activity(stop_selector, self.id)

        stop_selector = stop_selector[(self.get_elapsed() > 0)[stop_selector]]
        self.run_stop_callbacks(t, stop_selector)

    def register_stop_callbacks(self, func):
        self.__stop_callback.append(func)

    def register_start_callbacks(self, func):
        self.__start_callback.append(func)

    def __repr__(self):
        return f"Activity  {type(self)})"

    def run_stop_callbacks(self, t, stop_selector):
        for call_back in self.__stop_callback:
            call_back(self.id, t, stop_selector)

    def run_start_callbacks(self, t, start_selector):
        for call_bak in self.__start_callback:
            call_bak(self.id, t, start_selector)


class ActivityNone(ActivityPrimitive):
    id = 0

    def __init__(self, population):
        super().__init__(population)
        self.stationary = False
        # It usually gets added automatically so lets make sure it is the first one.
        self.id = 0


class ActivityList:
    def __init__(self, population):
        population_size = len(population)
        self.__index = np.arange(population_size)
        self.activity_values = np.zeros((population_size, len(ActivityProperties), 1), dtype=int)
        self.activities = [ActivityNone(population)]
        self.activity_types = [ActivityNone]

    def __repr__(self):
        return repr(self.activities)

    def __str__(self):
        return str(self.activities)

    def add(self, activity):
        if type(activity) in self.activity_types:
            activity.id = type(activity).id
            return

        self.activities.append(activity)
        self.activity_types.append(type(activity))
        self.activity_values = np.dstack([act.values for act in self.activities])
        activity.list = self
        setattr(activity.__class__, "id", len(self.activities) - 1)
        activity.id = len(self.activities) - 1

        # Connect activity values to the stack to free up memory and synchronize per activity operations with
        # ActivityList operations
        for ix, act in enumerate(self.activities):
            act.values = self.activity_values[:, :, ix]

    @property
    def shape(self):
        return self.activity_values.shape

    def __get_unique_activity_property(self, idx, prop_ix, activity_ids):
        """Given a list of ids, this method retrieves the activity property from the activities specified in
        activity_ids. The result has shape **n**x1 where n is the length of both activity_ids and ids"""
        pop_size = len(activity_ids)
        ids = self.__index[idx]
        if len(ids) == 0:
            return np.array([])

        if (activity_ids == -1).any():
            raise ValueError(f"Accessing properties for invalid activity id (id = -1). {activity_ids}")

        index_vector = np.array(tuple(zip(ids, [prop_ix] * pop_size, activity_ids)))
        index_vector = np.ravel_multi_index(index_vector.T, self.activity_values.shape)
        return self.activity_values.ravel()[index_vector]

    def __get_activity_property(self, idx, prop_ix, act_ids):
        if isinstance(idx, slice) or isinstance(idx, int):
            return self.activity_values[idx, prop_ix, act_ids]

        if isinstance(act_ids, slice) or isinstance(act_ids, int):
            return self.activity_values[idx, prop_ix, act_ids]

        if np.array(idx).shape == np.array(act_ids).shape:
            self.__get_unique_activity_property(idx, prop_ix, act_ids)

        return self.activity_values[idx, prop_ix, act_ids]

    def get_start(self, idx, act_ids) -> np.ndarray:
        return self.__get_activity_property(idx, ActivityProperties.start, act_ids)

    def get_duration(self, idx, act_ids) -> np.ndarray:
        return self.__get_activity_property(idx, ActivityProperties.duration, act_ids)

    def get_elapsed(self, idx, act_ids) -> np.ndarray:
        return self.__get_activity_property(idx, ActivityProperties.elapsed, act_ids)

    def get_accumulated(self, idx, act_ids) -> np.ndarray:
        return self.__get_activity_property(idx, ActivityProperties.accumulated, act_ids)

    def get_in_progress(self, idx, act_ids) -> np.ndarray:
        return self.__get_activity_property(idx, ActivityProperties.in_progress, act_ids)

    def get_blocked_for(self, idx, act_ids) -> np.ndarray:
        return self.__get_activity_property(idx, ActivityProperties.blocked_for, act_ids)

    def get_location(self, idx, act_ids) -> np.ndarray:
        return self.__get_activity_property(idx, ActivityProperties.location, act_ids)

    def __set_unique_activity_property(self, ids, prop_ix, activity, value):
        """Given a list of ids, this method sets the activity property from the activities specified in
                activity_ids to the value 'value'. The value is either a scalar or it has shape **n**x1,
                where n is the length of both activity_ids and ids"""
        pop_size = len(ids)
        index_vector = np.array(tuple(zip(ids, [prop_ix] * pop_size, activity)))
        index_vector = np.ravel_multi_index(index_vector.T, self.activity_values.shape)
        self.activity_values.ravel()[index_vector] = value

    def __set_activity_property(self, idx, prop_ix, act_ids, value):

        if isinstance(idx, slice):
            self.activity_values[idx, prop_ix, act_ids] = value
            return

        if isinstance(act_ids, slice):
            self.activity_values[idx, prop_ix, act_ids] = value
            return

        if np.array(idx).shape == np.array(act_ids).shape:
            self.__set_unique_activity_property(idx, prop_ix, act_ids, value)
            return

        self.activity_values[idx, prop_ix, act_ids] = value

    def set_start(self, idx, act_ids, value):
        self.__set_activity_property(idx, ActivityProperties.start, act_ids, value)

    def set_duration(self, idx, act_ids, value):
        self.__set_activity_property(idx, ActivityProperties.duration, act_ids, value)

    def set_elapsed(self, idx, act_ids, value):
        self.__set_activity_property(idx, ActivityProperties.elapsed, act_ids, value)

    def set_accumulated(self, idx, act_ids, value):
        self.__set_activity_property(idx, ActivityProperties.accumulated, act_ids, value)

    def set_in_progress(self, idx, act_ids, value):
        self.__set_activity_property(idx, ActivityProperties.in_progress, act_ids, value)

    def set_blocked_for(self, idx, act_ids, value):
        self.__set_activity_property(idx, ActivityProperties.blocked_for, act_ids, value)

    def set_location(self, idx, act_ids, value):
        return self.__set_activity_property(idx, ActivityProperties.location, act_ids, value)


class ActivityController(Protocol):
    z_order: int

    def get_new_activity(self, ids) -> (Any, Any):
        ...

    def has_new_activity(self, inactive_ids) -> np.ndarray:
        ...

    def registration_callback(self, ids):
        ...
