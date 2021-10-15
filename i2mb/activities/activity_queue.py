import numpy as np


class ActivityQueue:
    """FILO queue to control activity interruption. The queue removes the oldest activity if more activities are
    pushed into the queue"""
    def __init__(self, size, depth=3, padding=None):
        self.padding = padding
        self.size = size
        self.len = depth
        self.queue = np.full((size, depth), None, dtype=object)
        self.num_items = np.zeros(size, dtype=int)
        self.index = np.arange(size, dtype=int)

    def shift_right(self, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        self.queue[slice_, 1:] = self.queue[slice_, 0:-1]
        dropped_items = self.num_items[slice_] > self.len
        if dropped_items.any():
            self.num_items[slice_][dropped_items] = self.len

    def shift_left(self, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        self.queue[slice_, 0:-1] = self.queue[slice_, 1:]
        self.queue[slice_, -1] = self.padding

    def pop(self, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        response = self.queue[slice_, 0].copy()
        self.shift_left(slice_)
        self.num_items[slice_] -= 1
        empty = self.num_items[slice_] < 0
        if empty.any():
            self.num_items[slice_][empty] = 0

        return response

    def push(self, value, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        self.num_items[slice_] += 1
        self.shift_right(slice_)
        self.queue[slice_, 0] = value

    def reset(self):
        self.queue[:] = None
        self.num_items[:] = 0

    def __getitem__(self, item):
        return ActivityQueueView(item, self)

    def __str__(self):
        return str(self.queue)

    def push_end(self, value, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        # if the queue is full drop the incoming packet
        have_space = self.num_items[slice_] < self.len
        idx = self.index[slice_][have_space]
        self.queue[idx, self.num_items[idx]] = value
        self.num_items[slice_][have_space] += 1


class ActivityQueueView:
    def __init__(self, range_, queue: ActivityQueue):
        self.__queue = queue
        self.__range = range_

    @property
    def queue(self):
        return self.__queue.queue[self.__range]

    def push(self, value):
        self.__queue.push(value, self.__range)

    def push_end(self, value):
        self.__queue.push_end(value, self.__range)

    def pop(self):
        r = self.__queue.pop(self.__range)
        return r

    def reset(self):
        self.__queue.queue[self.__range, :] = None
        self.__queue.num_items[self.__range] = 0
