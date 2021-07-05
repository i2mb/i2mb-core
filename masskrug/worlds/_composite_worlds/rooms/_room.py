from masskrug.worlds import CompositeWorld
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
    def __init__(self, dims, origin, entry=0.5, rotation=0, scale=1, **kwargs):
        width, height = [dims[0], dims[1]] * scale
        x, y = origin[0], origin[1]

        offset = [0, 0, height, width, 0]  # array to compensate offset after rotation
        self.__rotation = rotation
        if rotation != 0 and rotation != 90 and rotation != 180 and rotation != 270:
            print("not a valid rotation")
            self.__rotation = 0
        rot = np.radians(self.rotation)

        # create door
        entry_offset = entry * width - DOOR_WIDTH * scale / 2
        entry_offset = max(min(0.985 * (width - DOOR_WIDTH * scale), entry_offset), 0)
        self.entry = np.array(
            [entry_offset * np.cos(rot) - height * np.sin(rot) + offset[int(self.rotation / 90) + 1],
             height * np.cos(rot) + entry_offset * np.sin(rot) + offset[int(self.rotation / 90)]])

        # define entry point where you leave and enter
        self.entry_point = [self.entry[0] + DOOR_WIDTH / 4 * (np.cos(rot) * 2.8 + np.sin(rot)),
                            self.entry[1] + DOOR_WIDTH / 4 * (np.sin(rot) * 2.8 - np.cos(rot))]

        self.door = Door(self.__rotation, self.entry, scale=scale)

        super().__init__(dims=dims, origin=origin, **kwargs)
        self.dims = self.dims * scale
        self.furniture = [self.door]

    @property
    def rotation(self):
        return self.__rotation

    @rotation.setter
    def rotation(self, rotation):
        if rotation != 0 and rotation != 90 and rotation != 180 and rotation != 270:
            print("not a valid rotation")
            self.__rotation = 0
        else:
            self.__rotation = rotation

    def draw_world(self, ax=None, origin=(0, 0), **kwargs):
        bbox = kwargs.get("bbox", False)
        self._draw_world(ax, origin=origin, **kwargs)
        for region in self.regions:
            region.draw_world(ax=ax, bbox=bbox, origin=origin + self.origin, **kwargs)


    def _draw_world(self, ax, bbox=False, origin=(0, 0), **kwargs):
        ax.add_patch(Rectangle(origin + self.origin, *self.dims, fill=False, linewidth=1.2, edgecolor='gray'))
        for f in self.furniture:
            f.draw(ax, bbox, origin + self.origin)

    def enter_world(self, n, idx=None, locations=None):
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

    def get_absolute_positions(self):
        abs_pos = np.zeros((len(self.population), 2))
        seen_index = np.array([])
        for r in self.regions:
            if r.is_empty():
                continue

            idx = r.population.index
            seen_index = np.union1d(seen_index, idx)
            abs_pos[idx] = r.population.position + r.origin

        idx_ = np.setdiff1d(self.population.index, seen_index)
        abs_pos[idx_] = self.population.position[idx_] + self.origin

        return abs_pos

    def get_furniture_grid(self):
        for id, f in enumerate(self.furniture):
            if id == 0:
                continue
            offset = [0, 0, f.length, f.width]
            self.furniture_origins[id - 1] = [f.origin[0] - offset[(int(f.rotation / 90) + 1) % 4],
                                              f.origin[1] - offset[(int(f.rotation / 90)) % 4]]
            self.furniture_upper[id - 1][0] = f.origin[0] + offset[(int(f.rotation / 90) + 3) % 4]
            self.furniture_upper[id - 1][1] = f.origin[1] + offset[(int(f.rotation / 90) + 2) % 4]
