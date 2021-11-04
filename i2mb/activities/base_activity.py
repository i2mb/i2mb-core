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
from typing import Union

import numpy as np

from i2mb.utils import Enumerator


def activity_vectorized(func):
    def __wrapper(x):
        return np.vectorize(func, otypes=[ActivityPrimitive])(x)

    return __wrapper


def enumerate_activity_indices(cls):
    # Making things explicit
    enumerator = Enumerator()
    cls.start_ix = enumerator.auto()
    cls.duration_ix = enumerator.auto()
    cls.elapsed_ix = enumerator.auto()
    cls.accumulated_ix = enumerator.auto()
    cls.in_progress_ix = enumerator.auto()
    cls.blocked_for_ix = enumerator.auto()


class ActivityPrimitive:
    # Vectorized property names
    __keys = ["start", "duration", "elapsed", "accumulated", "in_progress", "blocked_for"]

    # We need an class variable to hold the indexes, but they are assigned using the enumerate_activity_indices
    # function. We use that to maintain consistent enumeration between ActivityList and ActivityPrimitive.
    start_ix = None
    duration_ix = None
    elapsed_ix = None
    accumulated_ix = None
    in_progress_ix = None
    blocked_for_ix = None

    def __init__(self, population):
        # Where the activity will take place
        self.population = population

        # Rank determines the activities that can interrupt other activities, higher rank interrupt lower rank.
        self.rank = 0

        # Activity id this value gets updated once the activity is registered with the list.
        self.id = None

        n = len(self.population)
        self.location = np.full(n, None, dtype=object)

        # Device in location used during activity, e.g., bed
        self.device = np.full((n, 2), np.nan, dtype=float)

        self.stationary = True

        # Map properties into a contiguous 2D array
        self.__values = np.hstack([np.zeros((n, 1), dtype=int) for k in ActivityPrimitive.__keys])

        enumerate_activity_indices(ActivityPrimitive)

    def get_start(self) -> np.ndarray:
        return self.__values[:, self.start_ix]

    def get_duration(self) -> np.ndarray:
        return self.__values[:, self.duration_ix]

    def get_elapsed(self) -> np.ndarray:
        return self.__values[:, self.elapsed_ix]

    def get_accumulated(self) -> np.ndarray:
        return self.__values[:, self.accumulated_ix]

    def get_in_progress(self) -> np.ndarray:
        return self.__values[:, self.in_progress_ix]

    def get_blocked_for(self) -> np.ndarray:
        return self.__values[:, self.blocked_for_ix]

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

    def start_activity(self, t, start_activity):
        return

    def finalize_start(self, ids):
        if not hasattr(self.population, "position"):
            return

        device_selector = ~np.isnan(self.device).any(axis=1) & ids
        if device_selector.any():
            self.population.position[device_selector] = self.device[device_selector, :]

        if self.stationary:
            self.population.motion_mask[ids] = False

    def stop_activity(self, t, stop_selector):
        self.location[stop_selector] = None
        self.device[stop_selector, :] = np.nan
        if self.stationary and hasattr(self.population, "motion_mask"):
            self.population.motion_mask[stop_selector] = True

    def __repr__(self):
        return f"Activity  {type(self)})"


class ActivityNone(ActivityPrimitive):
    def __init__(self, population):
        super().__init__(population)
        self.stationary = False
        # It usually gets added automatically so lets make sure it is the first one.
        self.id = 0


class ActivityList:
    start_ix = None
    duration_ix = None
    elapsed_ix = None
    accumulated_ix = None
    in_progress_ix = None
    blocked_for_ix = None

    def __init__(self, population):
        population_size = len(population)
        self.population = population
        self.activity_values = np.empty((population_size, len(ActivityPrimitive.keys()), 1), dtype=object)
        self.activities = [ActivityNone(population)]
        self.activity_types = [ActivityNone]

        # Register vectorized property index getters
        enumerate_activity_indices(ActivityList)

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
        activity.id = len(self.activities) - 1

        # Connect activity values to the stack to free up memory and synchronize per activity operations with
        # ActivityList operations
        for ix, act in enumerate(self.activities):
            act.values = self.activity_values[:, :, ix]

    @property
    def shape(self):
        return self.activity_values.shape

    def get_activity_property(self, prop_ix, activity, ids: Union[np.ndarray, slice] = None):
        pop_size = len(activity)
        if ids is None:
            ids = slice(None)

        elif not isinstance(ids, slice) and ids.dtype is np.dtype(bool):
            assert len(ids) == self.activity_values.shape[0], ("If ids is a boolean array, the size should match the"
                                                               " population size.")
            ids = np.arange(pop_size)[ids]

        ids = self.population.index[ids]
        if len(ids) == 0:
            return np.array([])

        index_vector = np.array(tuple(zip(ids, [prop_ix] * pop_size, activity)))
        index_vector = np.ravel_multi_index(index_vector.T, self.activity_values.shape)
        return self.activity_values.ravel()[index_vector]

    def get_current_activity_property(self, prop_ix, ids=None):
        if ids is None:
            ids = slice(None)

        return self.get_activity_property(prop_ix, self.current_activity[ids], ids)

    def set_activity_property(self, prop_ix, value, activity, ids: Union[np.ndarray, slice] = None):
        pop_size = len(activity)
        if ids is None:
            pop_size = self.activity_values.shape[0]
            ids = slice(None)

        elif not isinstance(ids, slice) and ids.dtype is np.dtype(bool):
            assert len(ids) == self.activity_values.shape[0], ("If ids is a boolean array, the size should match the"
                                                               " population size.")
            if pop_size == 0:
                return

            ids = np.arange(pop_size)[ids]

        ids = self.population.index[ids]
        if len(ids) == 0:
            return

        index_vector = np.array(tuple(zip(ids, [prop_ix] * pop_size, activity)))
        index_vector = np.ravel_multi_index(index_vector.T, self.activity_values.shape)
        self.activity_values.ravel()[index_vector] = value

    def set_current_activity_property(self, prop_ix, value, ids=None):
        if ids is None:
            ids = slice(None)

        ids = self.population.index[ids]
        self.set_activity_property(prop_ix, value, self.current_activity[ids], ids=ids)

    def apply_descriptors(self, t, act_descriptors, ids=None):
        if ids is None:
            ids = slice(None)

        non_blocked_activities = self.get_activity_property(self.blocked_for_ix, act_descriptors[:, 0], ids) == 0
        # blocked_activities = act_descriptors[~non_blocked_activities, :]
        act_descriptors = act_descriptors[non_blocked_activities, :]
        ids = ids[non_blocked_activities]

        if len(ids) > 0:
            self.reset_current_activity(ids)
            self.stage_activity(act_descriptors, ids, t)

        return non_blocked_activities

    def stage_activity(self, act_descriptors, ids, t):
        self.stop_activities(t, ids)
        self.current_activity[ids] = act_descriptors[:, 0]
        act_descriptors[:, 1] = t
        self.set_current_activity_property(self.start_ix, act_descriptors[:, 1], ids)
        self.set_current_activity_property(self.duration_ix, act_descriptors[:, 2], ids)
        self.set_current_activity_property(self.blocked_for_ix, act_descriptors[:, 4], ids)

    def start_activities(self, ids):
        self.set_current_activity_property(self.in_progress_ix, 1, ids)

    def reset_current_activity(self, ids=None):
        if ids is None:
            ids = slice(None)

        self.set_current_activity_property(self.in_progress_ix, 0, ids)
        self.set_current_activity_property(self.start_ix, 0, ids)
        self.set_current_activity_property(self.duration_ix, 0, ids)
        self.set_current_activity_property(self.elapsed_ix, 0, ids)
        self.set_current_activity_property(self.blocked_for_ix, 0, ids)

    def stop_activities(self, t, stop_ids):
        if len(stop_ids) == 0:
            return

        self.reset_current_activity(stop_ids)
        for act in self.activities:
            stop_ids_ = stop_ids[(self.current_activity == act.id)[stop_ids]]
            if len(stop_ids_) > 0:
                act.stop_activity(t, stop_ids_)

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
