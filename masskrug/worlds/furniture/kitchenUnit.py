import numpy as np
from masskrug.worlds.furniture.base_furniture import BaseFurniture
from matplotlib.patches import Rectangle, PathPatch, Circle, FancyBboxPatch, Polygon
from matplotlib.path import Path


class Stove(BaseFurniture):
    plate_radius = 0.125

    def __init__(self, height=0.7, width=0.9, origin=None, rotation=0, scale=1):
        super().__init__(height=height, width=width, origin=origin, rotation=rotation, scale=scale)

        stove1_center = self.dims * np.array([0.25, 0.25])
        stove2_center = self.dims * np.array([0.25, 0.75])
        stove3_center = self.dims * np.array([0.75, 0.25])
        stove4_center = self.dims * np.array([0.75, 0.75])
        self.cook_plate_centers = [stove1_center, stove2_center, stove3_center, stove4_center]
        self.points.extend(self.cook_plate_centers)

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        # cook top
        # separators on the side
        ax.add_patch(Rectangle(abs_origin, self.width, self.height, fill=False,
                               linewidth=1.2, edgecolor='gray', alpha=0.4))

        # cooking plates
        radius = Stove.plate_radius
        for cp in self.cook_plate_centers:
            ax.add_patch(Circle(abs_origin + cp, radius, fill=True, linewidth=1.2,
                                facecolor="gray", edgecolor='gray', alpha=0.4))


class KitchenSink(BaseFurniture):
    def __init__(self, height=0.6, width=1.4, origin=None, rotation=0, scale=1):
        super().__init__(height=height, width=width, origin=origin, rotation=rotation, scale=scale)

        # Sink
        self.sink_corners = [self.dims * 0.05,
                             np.array([self.width * 0.475, self.height * 0.95])]
        self.points.extend(self.sink_corners)

        # Draining board
        self.draining_board_corners = [np.array([self.width * 0.525, self.height * 0.1]),
                                       np.array([self.width * 0.95, self.height * 0.9])]
        self.points.extend(self.draining_board_corners)

        # # Pattern
        db_width, db_height = self.draining_board_corners[1] - self.draining_board_corners[0]
        segment_height = 0.01 * db_height
        segment_width = 0.8 * db_width
        height_offset = segment_height / 2
        width_offset = (db_width - segment_width) / 2
        self.segment_origins = [np.array([width_offset, ((1. / 5 * i) * db_height) - height_offset]) +
                                self.draining_board_corners[0] for i in range(1, 5)]
        self.segment_corners = [o + [segment_width, segment_height] for o in self.segment_origins]
        self.points.extend(self.segment_corners)
        self.points.extend(self.segment_origins)

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        if bbox:
            ax.add_patch(Rectangle(abs_origin, self.width, self.height, fill=False,
                                   linewidth=1.2, edgecolor='blue', alpha=0.4))

        # sink
        # outer metal part
        width, height = self.dims
        ax.add_patch(FancyBboxPatch(abs_origin, width, height, boxstyle='round, pad=0, rounding_size=0.1',
                                    fill=False, linewidth=1.2, edgecolor='gray', alpha=0.4))
        #
        # inner sink
        width, height = abs(self.sink_corners[1] - self.sink_corners[0])
        local_origin = np.min(self.sink_corners, axis=0)
        ax.add_patch(FancyBboxPatch(abs_origin + local_origin, width, height,
                                    boxstyle='round, pad=0, rounding_size=0.1', mutation_scale=self.scale, fill=False,
                                    linewidth=1.2, edgecolor='gray', alpha=0.4))

        # draining board
        width, height = abs(self.draining_board_corners[1] - self.draining_board_corners[0])
        local_origin = np.min(self.draining_board_corners, axis=0)
        ax.add_patch(FancyBboxPatch(abs_origin + local_origin, width, height,
                                    boxstyle='round, pad=0, rounding_size=0.1', mutation_scale=self.scale, fill=False,
                                    linewidth=1.2, edgecolor='gray', alpha=0.4))

        # # pattern
        for orig, corner in zip(self.segment_origins, self.segment_corners):
            width, height = corner - orig
            ax.add_patch(
                Rectangle(abs_origin + orig, width, height, fill=False, facecolor="gray", linewidth=0.5, alpha=0.4,
                          edgecolor="gray"))


class KitchenUnit(BaseFurniture):
    def __init__(self, rotation=0, scale=1, origin=None, width=4, height=3, depth=1, shape='U'):
        super().__init__(height, width, origin, rotation, scale)
        self.depth = depth * scale

        self.stove = Stove()
        self.kitchen_sink = KitchenSink(origin=[1., 0.])
        self.add_equipment([self.stove, self.kitchen_sink])

        # make sure there is still enough space in between in case of a U-form and enough space for cooktop and sink in
        # case of a I-form
        if self.depth > self.width / 3:
            self.depth = self.width / 3

        self.counter_trim = []

        self.shape = shape
        self.build_kitchen(shape)

        self.points.extend(self.counter_trim)

    def build_kitchen(self, shape):
        if shape == "I":
            self.__build_i_shaped_kitchen()

        elif shape == "U":
            self.__build_u_shaped_kitchen()

        else:
            self.__build_l_shaped_kitchen()

    def __build_i_shaped_kitchen(self):
        # Create | shape profile
        self.counter_trim.append(np.array([self.depth, 0]))
        self.counter_trim.append(np.array([self.depth, self.height]))
        self.points.extend(self.counter_trim)

        # Position Kitchen Sink and stove
        spacing = (self.height - self.stove.width - self.kitchen_sink.width) / 3
        self.stove.origin = [self.depth / 2 - self.stove.height / 2, self.kitchen_sink.width + 2 * spacing]
        self.stove.rotate(90)

        self.kitchen_sink.origin = [self.depth / 2 - self.kitchen_sink.height / 2, spacing]
        self.kitchen_sink.rotate(90)

    def __build_u_shaped_kitchen(self):
        # Create U shape profile
        self.counter_trim.append(np.array([self.width, self.depth]))
        self.counter_trim.append(np.array([self.depth, self.depth]))
        self.counter_trim.append(np.array([self.depth, self.height - self.depth]))
        self.counter_trim.append(np.array([self.width * 0.75, self.height - self.depth]))
        self.counter_trim.append(np.array([self.width * 0.75, self.height]))
        self.points.extend(self.counter_trim)

        # Position Kitchen Sink and stove
        self.stove.origin = [self.depth / 2 - self.stove.height / 2, self.height / 2 - self.stove.width / 2]
        self.stove.rotate(90)

        self.kitchen_sink.origin = [self.width / 2 - self.kitchen_sink.width / 2, self.depth / 2 -
                                    self.kitchen_sink.height / 2]

    def __build_l_shaped_kitchen(self):
        # Create L shape profile
        self.counter_trim.append(np.array([self.width, self.depth]))
        self.counter_trim.append(np.array([self.depth, self.depth]))
        self.counter_trim.append(np.array([self.depth, self.height]))
        self.points.extend(self.counter_trim)

        # Position Kitchen Sink and stove
        self.stove.origin = [self.depth / 2 - self.stove.height / 2, self.height / 2 - self.stove.width / 2]
        self.stove.rotate(90)

        self.kitchen_sink.origin = [self.width / 2 - self.kitchen_sink.width / 2, self.depth / 2 -
                                    self.kitchen_sink.height / 2]

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin

        ax.add_patch(PathPatch(Path(abs_origin + self.counter_trim), fill=False,
                               linewidth=1.2, edgecolor='gray', alpha=0.4))

        for e in self.equipment:
            e.draw(ax, bbox=bbox, origin=abs_origin)
