from itertools import cycle

import numpy as np

from i2mb.activities.base_activity import ActivityNone


class ActivityDescriptor:
    """Agents can perform activities while in a location. The ActivityDescriptor class is a base class which describes
    where in the current location an activity can be performed."""
    def __init__(self, location=None, device=None, pos_id=None, interruptable=True, blocks_parent=False):
        # Where the activity will take place
        self.location = location
        self.location_id = location.id
        self.blocks_parent = blocks_parent

        # Device in location used during activity, e.g., bed
        self.device = device

        # Activity availability
        self.available = True

        # Keep track of how much this particular activity was used.
        self.accumulated = 0

        # Activity class associated to the descriptor
        self.activity_class = ActivityNone

        # Mark whether the activity is interruptable. An interruptable activity, will be resumed after the
        # interrupting activity finished.
        self.interruptable = interruptable

        # Specify the minimum duration an activity will take before is considered finished. Activities with minimum
        # duration will continue

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


class ActivityDescriptorSpecs:
    def __init__(self, act_idx=0, start=0, duration=0, priority_level=0, block_for=0, location_id=0,
                 blocks_location=0, blocks_parent_location=0, size=1):
        self.act_idx = self.align_variables(act_idx)
        self.start = self.align_variables(start)
        self.duration = self.align_variables(duration)
        self.priority_level = self.align_variables(priority_level)
        self.block_for = self.align_variables(block_for)
        self.location_id = self.align_variables(location_id)
        self.blocks_location = self.align_variables(blocks_location)
        self.blocks_parent_location = self.align_variables(blocks_parent_location)
        self.blocks_location |= self.blocks_parent_location.astype(bool)

        self.specifications = np.hstack([self.act_idx,
                                         self.start,
                                         self.duration,
                                         self.priority_level,
                                         self.block_for,
                                         self.location_id,
                                         self.blocks_location,
                                         self.blocks_parent_location])

        if size > 1:
            if len(self.specifications) > 1:
                raise RuntimeError("The length of specifications needs to be 1 for size to work.")

            self.specifications = np.tile(self.specifications, (size, 1))

    @staticmethod
    def align_variables(variable):
        if type(variable) is int:
            return np.array([variable]).reshape(-1, 1)

        return np.array(variable).reshape(-1, 1)


class ActivityDescriptorQueue:
    """Queues to control activity description. In FILO mode The queue removes the oldest activity if more activities are
    pushed into the queue. """
    num_descriptor_properties = 8
    empty_slot = -1

    def __init__(self, size, depth=3):
        self.size = size
        self.len = depth

        self.queue = np.full((size, ActivityDescriptorQueue.num_descriptor_properties, depth),
                             ActivityDescriptorQueue.empty_slot, dtype=int)

        self.num_items = np.zeros(size, dtype=int)
        self.index = np.arange(size, dtype=int)

        self.act_idx = self.queue[:, 0, 0]
        self.start = self.queue[:, 1, 0]
        self.duration = self.queue[:, 2, 0]
        self.priority_level = self.queue[:, 3, 0]
        self.block_for = self.queue[:, 4, 0]
        self.location_id = self.queue[:, 5, 0]
        self.blocks_location = self.queue[:, 6, 0]
        self.block_parent_location = self.queue[:, 7, 0]

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
        self.queue[:] = -1
        self.num_items[:] = 0

    def __getitem__(self, item):
        return ActivityDescriptorQueueView(item, self)

    def __str__(self):
        return str(self.queue)

    def append(self, value, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        if type(value) is ActivityDescriptorSpecs:
            value = value.specifications

        # if the queue is full drop the incoming packet
        have_space = self.num_items[slice_] < self.len
        idx = self.index[slice_][have_space]
        self.queue[idx, :, self.num_items[idx]] = value
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
        self.__queue.queue[self.__range, :, :] = 0
        self.__queue.num_items[self.__range] = 0


def create_null_descriptor_for_act_id(activity_ids):
    descriptor_array = np.zeros((len(activity_ids), ActivityDescriptorQueue.num_descriptor_properties), dtype=int)
    descriptor_array[:, 0] = activity_ids

    # Set location to -1, interpret as remain in place.
    descriptor_array[:, 5] = -1

    return descriptor_array


def convert_activities_to_descriptors(activity_ids, activity_view, current_location):
    descriptors = np.zeros((len(activity_ids), ActivityDescriptorQueue.num_descriptor_properties))
    descriptors[:, 0] = activity_ids
    # Duration
    descriptors[:, 2] = activity_view[:, ActivityNone.duration_ix] - activity_view[:, ActivityNone.elapsed_ix]

    # Priority level??
    descriptors[:, 3] = 0

    # blocking_for
    descriptors[:, 4] = activity_view[:, ActivityNone.blocked_for_ix]

    # location id
    descriptors[:, 5] = current_location
    return descriptors
