from collections import deque

import numpy as np

from i2mb.activities.activity_descriptors import CommuteCar
from i2mb.worlds import CompositeWorld


class Car(CompositeWorld):
    def __init__(self, orientation="vertical", **kwargs):
        # Positions are calculated by hand from the image
        self.seat_positions = np.array([[0.6, 0.5], [0.6, 1.5],
                                        [1.3, 0.4], [1.3, 1.], [1.3, 1.6]])

        self.seats = len(self.seat_positions)
        seats_idx = list(range(self.seats))
        np.random.shuffle(seats_idx)
        self.available_seats = deque(seats_idx)
        self.seat_assignments = {}

        self.orientation = 1
        kwargs["dims"] = (2., 4.)
        if orientation == "horizontal":
            self.orientation = 0
            kwargs["dims"] = (10.63, 2.55)

        super().__init__(**kwargs)

        if self.orientation == 1:
            self.seat_positions = self.seat_positions.dot([[0, 1], [-1, 0]]) + [self.dims[0], 0]

        activities = [CommuteCar(location=self)]
        self.local_activities.extend(activities)
        self.available_activities.extend(activities)
        self.default_activity = activities[0]

    def available_places(self):
        return len(self.available_seats)

    def sit_particles(self, idx):
        bool_idx = (self.population.index.reshape(-1, 1) == idx).any(axis=1)
        new_idx = np.arange(len(self.population))[bool_idx]
        new_assignments = {p: self.available_seats.popleft() for p in idx}
        self.seat_assignments.update(new_assignments)
        self.population.position[new_idx] = self.seat_positions[list(new_assignments.values()), :]

    def enter_world(self, n, idx=None, arriving_from=None):
        if hasattr(self.population, "motion_mask"):
            self.population.motion_mask[:] = False

        if idx is None:
            return super().enter_world(n)

        n = len(idx)
        seats = min(n, len(self.available_seats))
        sitting = idx[:seats]
        self.sit_particles(sitting)

        standing = n - seats

        if standing <= 0:
            idx_ = self.population.find_indexes(idx)
            return self.population.position[idx_]

        idx_ = self.population.find_indexes(idx[seats:])
        self.population.position[idx_] = np.random.random((standing, 2)) * (self.dims * 1 / 3) + (self.dims * 1 / 3)

    def exit_world(self, idx, global_population):
        bool_idx = (self.population.index.reshape(-1, 1) == idx).any(axis=1)
        if hasattr(self.population, "motion_mask"):
            self.population.motion_mask[bool_idx] = True

        for ix in idx:
            seat = self.seat_assignments.get(ix)
            if seat is not None:
                del self.seat_assignments[ix]
                self.available_seats.append(seat)
