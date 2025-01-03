from itertools import cycle
from numbers import Number

import numpy as np

from i2mb.activities import ActivityProperties, ActivityDescriptorProperties
from i2mb.utils.functions import callable_number


class ActivityDescriptor:
    __id_counter = 0
    """Agents can perform activities while in a location. The ActivityDescriptor class is a base class which describes
    where in the current location an activity can be performed."""
    def __init__(self, location=None, device=None, pos_id=None, interruptable=True, blocks_parent=False,
                 blocks_location=False, duration=0, blocks_for=0):
        # Where the activity will take place
        self.location = location
        self.location_index = location is not None and location.index or 0
        self.blocks_parent = blocks_parent
        self.blocks_location = blocks_location

        # Activity duration
        self.duration = duration
        if type(duration) is int:
            self.duration = callable_number(duration)

        # Period of time for which to block an activity
        self.blocks_for = blocks_for
        if type(blocks_for) is int:
            self.blocks_for = callable_number(blocks_for)

        # Device in location used during activity, e.g., bed
        self.device = device

        # if the device has multiple positions
        self.pos_id = pos_id

        # Unique descriptor identifier
        self.descriptor_id = self.get_id()

        # Activity availability
        self.available = True

        # Keep track of how much this particular activity was used.
        self.accumulated = 0

        # Activity class associated to the descriptor
        self.activity_class = None

        # Mark whether the activity is interruptable. An interruptable activity, will be resumed after the
        # interrupting activity finished.
        self.interruptable = interruptable

    def __repr__(self):
        return f"Activity  {type(self)}:(location:{self.location}, device:{self.device}, "\
                 f"pos_id:{self.pos_id})"

    def create_specs(self, size=1):
        location_index = self.location is not None and self.location.index or 0
        return ActivityDescriptorSpecs(act_idx=self.activity_class.id, duration=self.duration(size),
                                       block_for=self.blocks_for(size),
                                       location_ix=location_index,
                                       blocks_location=self.blocks_location,
                                       blocks_parent_location=self.blocks_parent,
                                       descriptor_id=self.descriptor_id,
                                       interruptable=self.interruptable, size=size)

    @staticmethod
    def get_id():
        ActivityDescriptor.__id_counter += 1
        return ActivityDescriptor.__id_counter


class CompoundActivityDescriptor:
    """Describes a sequence of activity descriptors. """
    def __init__(self, activity_descriptors):
        self.activity_descriptors = activity_descriptors
        self.__activity_cycle = cycle(self.activity_descriptors)

    def __next__(self):
        return next(self.__activity_cycle)


class ActivityDescriptorSpecs:
    def __init__(self, act_idx=0, start=0, duration=0, priority_level=0, block_for=0, location_ix=0,
                 blocks_location=0, blocks_parent_location=0, descriptor_id=-1, interruptable=1, size=1):
        self.size = size
        self.act_idx = self.align_variables(act_idx)
        self.start = self.align_variables(start)
        self.duration = self.align_variables(duration)
        self.priority_level = self.align_variables(priority_level)
        self.block_for = self.align_variables(block_for)
        self.location_ix = self.align_variables(location_ix)
        self.blocks_location = self.align_variables(blocks_location)
        self.blocks_parent_location = self.align_variables(blocks_parent_location)
        self.blocks_location |= self.blocks_parent_location.astype(bool)
        self.descriptor_id = self.align_variables(descriptor_id)
        self.interruptable = self.align_variables(interruptable)

        self.specifications = np.hstack([self.act_idx,
                                         self.start,
                                         self.duration,
                                         self.priority_level,
                                         self.block_for,
                                         self.location_ix,
                                         self.blocks_location,
                                         self.blocks_parent_location,
                                         self.interruptable,
                                         self.descriptor_id])

    def align_variables(self, variable):
        if isinstance(variable, Number):
            if self.size == 1:
                return np.array([variable], dtype=int).reshape(-1, 1)

            else:
                return np.full((self.size, 1), variable)

        return np.array(variable).reshape(-1, 1)

    @classmethod
    def merge_specs(cls, activity_specs):
        size = len(activity_specs)
        new_specs = cls(size=size)
        new_specs.specifications[:] = np.vstack([specs.specifications for specs in activity_specs])
        return new_specs


class ActivityDescriptorQueue:
    """Queues to control activity description. In FILO mode The queue removes the oldest activity if more activities are
    pushed into the queue. """
    empty_slot = -1

    def __init__(self, size, depth=3):
        self.size = size
        self.len = depth

        self.queue = np.full((size, len(ActivityDescriptorProperties), depth),
                             ActivityDescriptorQueue.empty_slot, dtype=int)

        self.num_items = np.zeros(size, dtype=int)
        self.index = np.arange(size, dtype=int)

        self.act_idx = self.queue[:, ActivityDescriptorProperties.act_idx.value, 0]
        self.start = self.queue[:, ActivityDescriptorProperties.start.value, 0]
        self.duration = self.queue[:, ActivityDescriptorProperties.duration.value, 0]
        self.priority_level = self.queue[:, ActivityDescriptorProperties.priority_level.value, 0]
        self.block_for = self.queue[:, ActivityDescriptorProperties.block_for.value, 0]
        self.location_ix = self.queue[:, ActivityDescriptorProperties.location_ix.value, 0]
        self.blocks_location = self.queue[:, ActivityDescriptorProperties.blocks_location.value, 0]
        self.block_parent_location = self.queue[:, ActivityDescriptorProperties.blocks_parent_location.value, 0]
        self.descriptor_id = self.queue[:, ActivityDescriptorProperties.descriptor_id.value, 0]

    def shift_right(self, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        self.queue[slice_, :, 1:] = self.queue[slice_, :, 0:-1]
        dropped_items = self.num_items[slice_] > self.len
        if dropped_items.any():
            self.num_items[slice_][dropped_items] = self.len

    def shift_left(self, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        self.queue[slice_, :, 0:-1] = self.queue[slice_, :, 1:]
        self.queue[slice_, :, -1] = ActivityDescriptorQueue.empty_slot

    def pop(self, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        if (self.num_items[slice_] == 0).any():
            raise ValueError("Trying to pop from an empty queue.")

        response = self.queue[slice_, :, 0].copy()
        self.shift_left(slice_)
        self.num_items[slice_] -= 1
        empty = self.num_items[slice_] < 0
        if empty.any():
            self.num_items[slice_][empty] = 0

        return response

    def push(self, value, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        if type(value) is ActivityDescriptorSpecs:
            value = value.specifications

        self.num_items[slice_] += 1
        self.shift_right(slice_)
        self.queue[slice_, :, 0] = value

    def reset(self):
        self.queue[:] = ActivityDescriptorQueue.empty_slot
        self.num_items[:] = 0

    def __getitem__(self, item):
        return ActivityDescriptorQueueView(item, self)

    def __str__(self):
        return str(self.queue)

    def append(self, value, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        if type(slice_) is int:
            slice_ = [slice_]

        if type(value) is ActivityDescriptorSpecs:
            value = value.specifications

        # if the queue is full drop the incoming packet
        have_space = self.num_items[slice_] < self.len
        idx = self.index[slice_][have_space]
        if len(value) == 1:
            self.queue[idx, :, self.num_items[idx]] = value
        else:
            self.queue[idx, :, self.num_items[idx]] = value[have_space]
        self.num_items[idx] += 1


class ActivityDescriptorQueueView:
    def __init__(self, range_, queue: ActivityDescriptorQueue):
        self.__queue = queue
        self.__range = range_

    @property
    def queue(self):
        return self.__queue.queue[self.__range]

    def push(self, value):
        self.__queue.push(value, self.__range)

    def append(self, value):
        self.__queue.append(value, self.__range)

    def pop(self):
        r = self.__queue.pop(self.__range)
        return r

    def reset(self):
        self.__queue.queue[self.__range, :, :] = ActivityDescriptorQueue.empty_slot
        self.__queue.num_items[self.__range] = 0


def create_null_descriptor_for_act_id(activity_ids):
    descriptor_array = np.zeros((len(activity_ids), len(ActivityDescriptorProperties)), dtype=int)
    descriptor_array[:, ActivityDescriptorProperties.act_idx.value] = activity_ids

    # Set location to index 0 of the relocator region_index, which is interpreted as remain in place.
    descriptor_array[:, ActivityDescriptorProperties.location_ix.value] = 0

    # Descriptor id
    descriptor_array[:, ActivityDescriptorProperties.descriptor_id.value] = ActivityDescriptorQueue.empty_slot

    return descriptor_array


def convert_activities_to_descriptors(activity_ids, activity_view, current_location, descriptor_ids):
    descriptors = np.zeros((len(activity_ids), len(ActivityDescriptorProperties)))
    descriptors[:, ActivityDescriptorProperties.act_idx] = activity_ids

    # Duration
    descriptors[:, ActivityDescriptorProperties.duration] = (
            activity_view[:, ActivityProperties.duration] -
            activity_view[:, ActivityProperties.elapsed])

    # Priority level??
    descriptors[:, ActivityDescriptorProperties.priority_level] = 0

    # blocking_for
    descriptors[:, ActivityDescriptorProperties.block_for] = activity_view[:, ActivityProperties.blocked_for]

    # location id
    descriptors[:, ActivityDescriptorProperties.location_ix] = current_location

    # # Descriptor id
    # descriptors[:, ActivityDescriptorProperties.descriptor_id] = descriptor_ids
    return descriptors
