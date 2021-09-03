import numpy as np


class ActivityQueueView:
    def __init__(self, range_, queue):
        self.__queue = queue
        self.__range = range_

    @property
    def queue(self):
        return self.__queue.queue[self.__range]

    def push(self, value):
        self.__queue.shift_right(self.__range)

        range_ = self.__range
        if isinstance(range_, tuple):
            range_ = range_[0]
        self.__queue.queue[range_, 0] = value

    def pop(self):
        range_ = self.__range
        if isinstance(range_, tuple):
            range_ = range_[0]

        r = self.__queue.queue[range_, 0].copy()
        self.__queue.shift_left(self.__range)
        return r

    def reset(self):
        self.__queue.queue[self.__range] = None


class ActivityQueue:
    """FILO queue to control activity interruption. The queue removes the oldest activity if more activities are
    pushed into the queue"""
    def __init__(self, size, depth=3, padding=None):
        self.padding = padding
        self.size = size
        self.len = depth
        self.queue = np.full((size, depth), None, dtype=object)

    def shift_right(self, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        self.queue[slice_, 1:] = self.queue[slice_, 0:-1]

    def shift_left(self, slice_=None):
        if slice_ is None:
            slice_ = slice(None)

        self.queue[slice_, 0:-1] = self.queue[slice_, 1:]
        self.queue[slice_, -1] = self.padding

    def pop(self):
        response = self.queue[:, 0].copy()
        self.shift_left()
        return response

    def push(self, value):
        self.shift_right()
        self.queue[:, 0] = value

    def reset(self):
        self.queue[:] = None

    def __getitem__(self, item):
        return ActivityQueueView(item, self)

    def __str__(self):
        return str(self.queue)
