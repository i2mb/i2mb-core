from matplotlib.patches import Rectangle, PathPatch, Path
from i2mb.worlds import CompositeWorld
from i2mb.utils import global_time
from collections import deque
import numpy as np

"""
    :param scale: percentage value for lift scaling
    :type scale: float, optional
    :param width: width of lift
    :type width: float, optional
    :param height: height of building
    :type height: float, optional
    :param occupation: number of people possible in lift
    :type occupation: int, optional
"""


class Lift(CompositeWorld):
    def __init__(self, scale=1, width=1.5, height=15.5, occupation=6, **kwargs):
        super().__init__(dims=(width, height), **kwargs)
        self.dims = self.dims * scale
        self.floor_number = 0
        self.occupied = np.zeros((occupation,), dtype=bool)
        self.lift_width = 0.8 * self.dims[0]
        self.lift_height = self.dims[0] * 1.6
        self.lift_origin = [0.1 * self.dims[0], self.dims[1] / 2 - self.dims[0] * 0.8]
        self.seats = [[self.lift_origin[0] + self.lift_width / 4,
                       self.lift_origin[1] + (i + 1) * self.lift_height / (occupation // 2 + 1)] for i in
                      range(occupation // 2)]
        j = occupation // 2 + occupation % 2
        self.seats += [
            [self.lift_origin[0] + self.lift_width * 3 / 4, self.lift_origin[1] + (i + 1) * self.lift_height / (j + 1)]
            for i in range(j)]

        self.available_seats = deque(self.seats, )
        self.queue = deque()
        self.travel_time = -1

    '''
    moves lift to next floor_number in queue
    :param t: current time step
    :type t: int
    '''

    def move_lift(self, t):
        x = self.queue.popleft()
        self.floor_number = x
        if len(self.queue) > 0:
            self.get_time(self.queue[0], t)
        else:
            self.travel_time = -1

    '''
    appends floor_number to lift queue
    :param floor_number: floor_number lift has to move to
    :type floor_number: int 
    '''

    def call_lift(self, floor_number):
        for f in floor_number:
            if f in self.queue:
                continue
            if len(self.queue) > 0:
                if (self.queue[0] < f < self.floor_number) or (self.floor_number < f < self.queue[0]):
                    self.queue.insert(0, f)
                    continue
            self.queue.append(f)

    '''
    calculates travel time from current floor to new_floor
    :param new_floor: next floor_number 
    :type new_floor: int
    :param t: current time step
    :type t: int
    returns time in sec
    '''

    def get_time(self, new_floor, t):
        # travel time 4 secs : 1 sec per meter
        self.travel_time = abs(new_floor - self.floor_number) * global_time.make_time(minutes=1) / 15 + t

    def draw_world(self, ax=None, origin=(0, 0), **kwargs):
        bbox = kwargs.get("bbox", False)
        self._draw_world(ax, origin=origin, **kwargs)
        for region in self.regions:
            region.draw_world(ax=ax, origin=origin + self.origin, **kwargs)

    def _draw_world(self, ax, bbox=True, origin=(0, 0), **kwargs):
        abs_origin = origin + self.origin
        ax.add_patch(Rectangle(abs_origin, self.dims[0], self.dims[1], fill=False, linewidth=1.2, edgecolor="gray"))
        ax.add_patch(
            Rectangle(abs_origin + self.lift_origin,
                      self.lift_width, self.lift_height,
                      fill=True, facecolor="blue", linewidth=1.2,
                      alpha=0.2, edgecolor="blue"))
        ax.add_patch(
            PathPatch(Path([(abs_origin[0] + self.dims[0] / 2, abs_origin[1] + self.dims[1] / 2 + self.dims[0] * 0.8),
                            (abs_origin[0] + self.dims[0] / 2, abs_origin[1] + self.dims[1])]),
                      fill=True, facecolor="blue", linewidth=1.2, alpha=0.2, edgecolor="blue"))

    def step(self, t):
        if len(self.queue) == 0:
            return
        if self.travel_time < 0:
            self.get_time(self.queue[0], t)
        if t < self.travel_time:
            return
        self.move_lift(t)
        return
