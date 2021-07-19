from masskrug.worlds import CompositeWorld
from masskrug.worlds._area import Area
from masskrug.worlds.furniture.door import Door, DOOR_WIDTH
from matplotlib.patches import Rectangle, Arc
import numpy as np

"""
    :param entry: Position of the door in percentage of room width.
    :type entry: float, optional
    :param rotation: Angle of counterclockwise rotation of the room to fit into an apartment. Measured in degrees.
        Values other than 0, 90, 180 and 270 are ignored.
    :type rotation: int, optional
    :param scale: percentage value for room scaling
    :type scale: float, optional
"""


class BaseRoom(CompositeWorld):
    def __init__(self, dims, entry=0.5, scale=1, **kwargs):
        self.furniture = []

        super().__init__(dims=dims, scale=scale, **kwargs)
        self.register_sub_areas(self.furniture)

        width, height = self.dims

        # create door
        entry_offset = entry * height - DOOR_WIDTH * scale / 2
        # entry_offset = max(min(0.985 * (width - DOOR_WIDTH * scale), entry_offset), 0)
        self.entry = np.array([width - DOOR_WIDTH * scale, entry_offset])
        # self.entry = self.rotate_point(self.entry)

        # define entry point where you leave and enter
        self.entry_point = self.entry.copy() + (0.5*DOOR_WIDTH, 0.5*DOOR_WIDTH)
        self.points.append(self.entry_point)

        self.door = Door(0, self.entry, scale=scale)
        self.furniture.append(self.door)
        self.points.extend([p for f in self.furniture for p in [f.origin, f.opposite]])
        self.furniture_origins = []
        self.furniture_upper = []

    def add_furniture(self, new_furniture):
        self.points.extend([p for f in new_furniture for p in [f.origin, f.opposite]])
        self.furniture.extend(new_furniture)
        self.furniture_origins.extend([f.origin for f in new_furniture])
        self.furniture_origins.extend([f.opposite for f in new_furniture])

    def draw_world(self, ax=None, origin=(0, 0), **kwargs):
        bbox = kwargs.get("bbox", False)
        self._draw_world(ax, origin=self.origin + origin, **kwargs)
        for region in self.regions:
            region.draw_world(ax=ax, bbox=bbox, origin=origin + self.origin, **kwargs)

        for f in self.furniture:
            f.draw(ax, bbox, origin + self.origin)

    def enter_world(self, n, idx=None, arriving_from=None):
        return [self.entry_point] * n

    def exit_world(self, idx):
        bool_ix = self.population.find_indexes(idx)
        self.population.motion_mask[bool_ix] = True

    def check_positions(self, mask):
        for r in self.regions:
            if r.is_empty():
                continue
            r_mask = mask[r.population.index]
            r.check_positions(r_mask)

        if self.is_empty():
            return

        positions = self.population.position[self.location == self]
        mask = mask[self.location == self]
        # check if inside room
        check_top = mask.ravel() & (positions[:, 0] > self.dims[0])
        check_right = mask.ravel() & (positions[:, 1] > self.dims[1])
        check_left = mask.ravel() & (positions[:, 0] < 0)
        check_bottom = mask.ravel() & (positions[:, 1] < 0)

        positions[check_top, 0] = self.dims[0]
        positions[check_right, 1] = self.dims[1]
        positions[check_left, 0] = 0
        positions[check_bottom, 1] = 0

        ids = self.population.index[self.location == self]

        # check if on furniture
        checked = np.zeros(len(positions), dtype=bool)
        if self.furniture_origins is not None:
            for origin, upper in zip(self.furniture_origins, self.furniture_upper):
                if checked.all():
                    break
                temp1 = origin <= positions
                over_origin = np.logical_and(temp1[:, 0], temp1[:, 1]).ravel()
                temp2 = upper >= positions
                under_upper = np.logical_and(temp2[:, 0], temp2[:, 1]).ravel()

                on_furniture = over_origin & under_upper
                x0, y0 = origin
                x1, y1 = upper

                for id, pos in zip(ids[on_furniture & mask.ravel()], positions[on_furniture & mask.ravel()]):
                    rectangle = np.array([origin, upper, np.array([x0, y1]), np.array([x1, y0])])
                    distances = np.array([pos - origin, upper - pos, pos - rectangle[2], rectangle[3] - pos])
                    mag = lambda x: np.sqrt(sum(i ** 2 for i in x))
                    abs_diff = np.array([mag(d) for d in distances])
                    min = distances[0:2].argmin()
                    dmin = abs_diff[~np.isclose(abs_diff, 0)].argmin()
                    idx = np.argwhere(ids == id).ravel()[0]
                    positions[idx, min % 2] = origin[min % 2] if min // 2 == 0 else upper[min % 2]

                    if not hasattr(self.population, "target"):
                        continue

                    tolerance = 0.01
                    target = self.population.target[idx]
                    particle_is_left = pos[0] <= (x0 + x1) / 2
                    particle_is_under = pos[1] <= (y0 + y1) / 2
                    on_the_right_side = ((target[0] <= x0 + tolerance) & particle_is_left) | \
                                        ((target[0] >= x1 - tolerance) & ~particle_is_left) | \
                                        ((target[1] <= y0 + tolerance) & particle_is_under) | \
                                        ((target[1] >= y1 - tolerance) & ~particle_is_under)
                    walk_around = np.isnan(self.population.inter_target[idx]).any() & \
                                  ~(np.isnan(target).any()) & ~on_the_right_side
                    if walk_around:
                        points = rectangle[abs_diff != 0]
                        self.population.inter_target[idx] = points[dmin]

                checked[on_furniture] = True

        self.population.position[self.location == self] = np.array(positions)

    def get_furniture_grid(self):
        for id, f in enumerate(self.furniture):
            if id == 0:
                continue

            offset = [0, 0, f.height, f.width]
            self.furniture_origins[id - 1] = [f.origin[0] - offset[(int(f.rotation / 90) + 1) % 4],
                                              f.origin[1] - offset[(int(f.rotation / 90)) % 4]]
            self.furniture_upper[id - 1][0] = f.origin[0] + offset[(int(f.rotation / 90) + 3) % 4]
            self.furniture_upper[id - 1][1] = f.origin[1] + offset[(int(f.rotation / 90) + 2) % 4]
