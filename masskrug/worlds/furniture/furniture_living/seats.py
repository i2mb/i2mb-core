import numpy as np

from matplotlib.patches import Rectangle


class Sofa:
    def __init__(self, rotation=0, origin=None, width=2.2, length=1, scale=1):
        self.width = width * scale
        self.length = length * scale
        self.__origin = origin
        self.rotation = rotation
        if origin is None:
            self.__origin = np.array([0, 0])
        rot = np.radians(self.rotation)
        self.sleeping_pos = []
        self.sleeping_target = []
        self.sitting_pos = []
        self.sitting_target = []
        self.set_sitting_position()
        self.num_seats = 2
        self.occupants = 0

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)

    def get_sitting_position(self):
        return self.origin + self.sitting_pos

    def get_sitting_target(self):
        return self.origin + self.sitting_target

    def set_sitting_position(self):
        rot = np.radians(self.rotation)

        self.sitting_pos = [[i * np.cos(rot) * self.width / 4 - np.sin(rot) * self.length / 1.5, \
                             i * np.sin(rot) * self.width / 4 + np.cos(rot) * self.length / 1.5] for i in
                            range(1, 4, 2)]
        self.sitting_target = [[i * np.cos(rot) * self.width / 4 - np.sin(rot) * self.length, \
                                i * np.sin(rot) * self.width / 4 + np.cos(rot) * self.length] for i in
                               range(1, 4, 2)]

    def get_sleeping_position(self):
        return self.origin + self.sleeping_pos

    def get_sleeping_target(self):
        return self.origin + self.sleeping_target

    def set_sleeping_position(self):
        rot = np.radians(self.rotation)

        self.sleeping_pos = [np.cos(rot) * self.width / 2 - np.sin(rot) * self.length / 1.5,
                             np.sin(rot) * self.width / 2 + np.cos(rot) * self.length / 1.5]
        self.sleeping_target = [np.cos(rot) * self.width / 2 - np.sin(rot) * self.length,
                                np.sin(rot) * self.width / 2 + np.cos(rot) * self.length]

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        rot = np.radians(self.rotation)
        offset = [np.cos(rot) * self.width * 0.1, np.sin(rot) * self.width * 0.1]
        # armrest
        ax.add_patch(Rectangle(abs_origin, self.width * 0.1, self.length * 1.05, self.rotation,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))
        ax.add_patch(Rectangle((abs_origin[0] + offset[0] * 9, abs_origin[1] + offset[1] * 9),
                               self.width * 0.1, self.length * 1.05, self.rotation,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))
        # backrest
        ax.add_patch(Rectangle((abs_origin + offset), self.width * 0.8, self.width * 0.1, self.rotation,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))
        # seat squab
        ax.add_patch(Rectangle((abs_origin[0] + offset[0] - np.sin(rot) * self.width * 0.1,
                                abs_origin[1] + offset[1] + np.cos(rot) * self.width * 0.1),
                               self.width * 0.8, self.length - self.width * 0.1, self.rotation,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))


class Armchair:
    def __init__(self, rotation=0, origin=None, width=1.2, length=1, scale=1):
        self.width = width * scale
        self.length = length * scale
        self.__origin = origin
        self.rotation = rotation
        if origin is None:
            self.__origin = np.array([0, 0])
        rot = np.radians(self.rotation)
        self.sitting_pos = []
        self.sitting_target = []
        self.set_sitting_position()
        self.num_seats = 1
        self.occupants = 0

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)

    def set_sitting_position(self):
        rot = np.radians(self.rotation)
        self.sitting_pos = np.cos(rot) * self.width / 2 - np.sin(rot) * self.length / 1.5, \
                           np.sin(rot) * self.width / 2 + np.cos(rot) * self.length / 1.5
        self.sitting_target = np.cos(rot) * self.width / 2 - np.sin(rot) * self.length, \
                              np.sin(rot) * self.width / 2 + np.cos(rot) * self.length

    def get_sitting_position(self):
        return [self.origin + self.sitting_pos]

    def get_sitting_target(self):
        return [self.origin + self.sitting_target]

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        rot = np.radians(self.rotation)
        offset = [np.cos(rot) * self.width * 0.2, np.sin(rot) * self.width * 0.2]
        # armrest
        ax.add_patch(Rectangle(abs_origin, self.width * 0.2, self.length * 1.05, self.rotation,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))
        ax.add_patch(Rectangle((abs_origin[0] + offset[0] * 4, abs_origin[1] + offset[1] * 4),
                               self.width * 0.2, self.length * 1.05, self.rotation,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))
        # backrest
        ax.add_patch(Rectangle((abs_origin + offset), self.width * 0.6, self.width * 0.2, self.rotation, fill=False,
                               facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))
        # seat squab
        ax.add_patch(Rectangle((abs_origin[0] + offset[0] - np.sin(rot) * self.width * 0.2,
                                abs_origin[1] + offset[1] + np.cos(rot) * self.width * 0.2),
                               self.width * 0.6, self.length - self.width * 0.2, self.rotation,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))
