import numpy as np

from masskrug.utils import global_time
from masskrug.worlds import CompositeWorld


class House(CompositeWorld):
    def __init__(self, gain=5, always_on=False, use_beds=False, num_rooms=4, **kwargs):

        self.num_rooms = num_rooms
        self.beds = np.zeros((self.num_rooms * 4 + 1, 2))
        CompositeWorld.__init__(self, **kwargs)
        self.use_beds = use_beds
        self.always_on = always_on
        self.gain = gain
        self.dinner_table = self.dims / 2
        self.activation_times = [7, 18, 19, 20]
        self.bed_assignment = None

    @CompositeWorld.dims.setter
    def dims(self, value):
        CompositeWorld.dims.fset(self, value)
        self.dinner_table = self.dims / 2 + self.origin

        vertical = int(self.dims[1] > self.dims[0])
        num_rooms_vertical = self.dims[vertical] // 2
        num_rooms_horizontal = self.dims[int(not vertical)] // 3
        room_height = self.dims[vertical] / num_rooms_vertical
        room_width = self.dims[int(not vertical)] / num_rooms_horizontal

        room_origins = np.zeros((self.num_rooms, 2))
        for room in range(len(room_origins)):
            room_origins[room, :] = [(room_height * room) % self.dims[vertical],
                                     (room_width * ((room_height * room) // self.dims[vertical]))][
                                    ::(-1) ** (2 + vertical)]

        bed_locs = [
            [0.35, 0.75],
            [1.65, 0.75],
            [0.35, 1.25],
            [1.65, 1.25]
        ]
        self.beds[0] = [1 - 0.40, 1.][::(-1) ** (2 + vertical)]
        self.beds[1] = [1 + 0.40, 1.][::(-1) ** (2 + vertical)]
        for bed in range(2, (len(self.beds))):
            room = 1 + (bed - 2) % (self.num_rooms - 1)
            bed_loc = ((bed - 2) // (self.num_rooms - 1)) % (self.num_rooms - 1)
            self.beds[bed] = room_origins[room] + bed_locs[bed_loc][::(-1) ** (2 + vertical)]

    @CompositeWorld.origin.setter
    def origin(self, value):
        CompositeWorld.origin.fset(self, value)

    def distance(self):
        return self.population.position - self.dinner_table

    def assign_beds(self):
        self.beds = self.beds[:len(self.population)]
        self.bed_assignment = self.population.index.copy().reshape(-1, 1)

    def step(self, t):
        if not self.population:
            return

        n = len(self.population)

        if hasattr(self.population, "sleep"):
            # Wake people up
            wake_up = (~self.population.sleep & self.population.in_bed).ravel()
            if wake_up.any():
                self.population.position[wake_up] = self.enter_world(wake_up.sum())
                self.population.in_bed[wake_up] = False
                self.population.motion_mask[wake_up] = True

            # Send people to sleep
            send_to_bed = (self.population.sleep & ~self.population.in_bed).ravel()
            if send_to_bed.any():
                self.population.in_bed[send_to_bed] = True
                self.population.motion_mask[send_to_bed] = False
                beds = (self.bed_assignment == self.population.index).any(axis=1)
                self.population.position[send_to_bed] = self.beds[beds][send_to_bed]

        hour = global_time.hour(t)
        if not self.always_on and hour not in self.activation_times:
            self.population.gravity[:] = np.zeros((n, 2))
            return

        dist_x, dist_y = self.distance().T
        mag_2 = dist_x ** 2 + dist_y ** 2

        self.population.gravity[:] = list(zip((1 / (mag_2 + 100) * np.sign(dist_x) * -self.gain),
                                              (1 / (mag_2 + 100) * np.sign(dist_y) * -self.gain)))

        return
