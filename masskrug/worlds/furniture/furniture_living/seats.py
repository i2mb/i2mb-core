import numpy as np
from masskrug.worlds.furniture.base_furniture import BaseFurniture

from matplotlib.patches import Rectangle


class Sofa(BaseFurniture):
    def __init__(self, rotation=0, origin=None, width=2.2, height=1, scale=1):
        super().__init__(height, width, origin, rotation, scale)

        self.sleeping_pos = []
        self.sleeping_target = []
        self.sitting_pos = []
        self.sitting_target = []
        self.set_sitting_position()
        self.num_seats = 2
        self.occupants = 0

        self.arm_rest_left_origin = np.array([0.0, 0.0])
        self.arm_rest_left_opposite = np.array([width * 0.1, height * 1.05])
        self.arm_rest_right_origin = np.array([width * 0.9, 0.])
        self.arm_rest_right_opposite = np.array([width, height * 1.05])
        self.back_rest_origin = np.array([width * 0.1, 0.0])
        self.back_rest_opposite = np.array([width * 0.9, width * 0.1])
        self.generate_sofa_parts()

    def generate_sofa_parts(self):
        self.points.extend([self.arm_rest_left_origin, self.arm_rest_left_opposite])
        self.points.extend([self.arm_rest_right_origin, self.arm_rest_right_opposite])
        self.points.extend([self.back_rest_origin, self.back_rest_opposite])

    def get_sitting_position(self):
        if not self.sitting_pos:
            return self.sitting_pos

        return self.origin + self.sitting_pos

    def get_sitting_target(self):
        if not self.sitting_target:
            return self.sitting_target

        return self.origin + self.sitting_target

    def set_sitting_position(self):
        rot = np.radians(self.rotation)

        self.sitting_pos = [[i * np.cos(rot) * self.width / 4 - np.sin(rot) * self.height / 1.5,
                             i * np.sin(rot) * self.width / 4 + np.cos(rot) * self.height / 1.5] for i in
                            range(1, 4, 2)]
        self.sitting_target = [[i * np.cos(rot) * self.width / 4 - np.sin(rot) * self.height,
                                i * np.sin(rot) * self.width / 4 + np.cos(rot) * self.height] for i in
                               range(1, 4, 2)]

    def get_sleeping_position(self):
        return self.origin + self.sleeping_pos

    def get_sleeping_target(self):
        return self.origin + self.sleeping_target

    def set_sleeping_position(self):
        rot = np.radians(self.rotation)

        self.sleeping_pos = [np.cos(rot) * self.width / 2 - np.sin(rot) * self.height / 1.5,
                             np.sin(rot) * self.width / 2 + np.cos(rot) * self.height / 1.5]
        self.sleeping_target = [np.cos(rot) * self.width / 2 - np.sin(rot) * self.height,
                                np.sin(rot) * self.width / 2 + np.cos(rot) * self.height]

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        if bbox:
            ax.add_patch(Rectangle(abs_origin, self.width, self.height,
                                   fill=True, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))

        # armrest left
        width, length = self.arm_rest_left_opposite - self.arm_rest_left_origin
        ax.add_patch(Rectangle(abs_origin + self.arm_rest_left_origin, width, length,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))

        # armrest right
        width, length = self.arm_rest_right_opposite - self.arm_rest_right_origin
        ax.add_patch(Rectangle(abs_origin + self.arm_rest_right_origin, width, length,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))

        # backrest
        width, length = self.back_rest_opposite - self.back_rest_origin
        ax.add_patch(Rectangle(abs_origin + self.back_rest_origin, width, length,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))

        # seat cushion
        ax.add_patch(Rectangle(abs_origin, self.width, self.height,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))


class Armchair(BaseFurniture):
    def __init__(self, rotation=0, origin=None, width=1.2, height=1, scale=1):
        super().__init__(height, width, rotation, scale)
        self.origin = origin
        self.sitting_pos = []
        self.sitting_target = []
        self.set_sitting_position()
        self.num_seats = 1
        self.occupants = 0

        self.arm_rest_left_origin = np.array([0.0, 0.0])
        self.arm_rest_left_opposite = np.array([width * 0.15, height * 1.05])
        self.arm_rest_right_origin = np.array([width * 0.85, 0.])
        self.arm_rest_right_opposite = np.array([width, height * 1.05])
        self.back_rest_origin = np.array([width * 0.15, 0.0])
        self.back_rest_opposite = np.array([width * 0.85, width * 0.15])
        self.generate_sofa_parts()

    def generate_sofa_parts(self):
        self.points.extend([self.arm_rest_left_origin, self.arm_rest_left_opposite])
        self.points.extend([self.arm_rest_right_origin, self.arm_rest_right_opposite])
        self.points.extend([self.back_rest_origin, self.back_rest_opposite])

    def set_sitting_position(self):
        rot = np.radians(self.rotation)
        self.sitting_pos = np.cos(rot) * self.width / 2 - np.sin(rot) * self.height / 1.5, \
                           np.sin(rot) * self.width / 2 + np.cos(rot) * self.height / 1.5
        self.sitting_target = np.cos(rot) * self.width / 2 - np.sin(rot) * self.height, \
                              np.sin(rot) * self.width / 2 + np.cos(rot) * self.height

    def get_sitting_position(self):
        return [self.origin + self.sitting_pos]

    def get_sitting_target(self):
        return [self.origin + self.sitting_target]

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        # ax.add_patch(Rectangle(abs_origin, self.width, self.height,
        #                        fill=True, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))

        # armrest left
        width, length = self.arm_rest_left_opposite - self.arm_rest_left_origin
        ax.add_patch(Rectangle(abs_origin + self.arm_rest_left_origin, width, length,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))

        # armrest right
        width, length = self.arm_rest_right_opposite - self.arm_rest_right_origin
        ax.add_patch(Rectangle(abs_origin + self.arm_rest_right_origin, width, length,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))

        # backrest
        width, length = self.back_rest_opposite - self.back_rest_origin
        ax.add_patch(Rectangle(abs_origin + self.back_rest_origin, width, length,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))

        # seat cushion
        ax.add_patch(Rectangle(abs_origin, self.width, self.height,
                               fill=False, facecolor="gray", linewidth=1.2, alpha=0.4, edgecolor="gray"))
