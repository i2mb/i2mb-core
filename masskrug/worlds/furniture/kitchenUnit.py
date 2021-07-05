import numpy as np
from matplotlib.patches import Rectangle, PathPatch, Circle, FancyBboxPatch, Polygon
from matplotlib.path import Path


class KitchenUnit:
    def __init__(self, rotation=0, scale=1, origin=None, width=4, length=3, depth=1, shape='U'):
        self.width = width
        self.length = length
        self.depth = depth * scale
        self.scale = scale
        # make sure there is still enough space in between in case of a U-form and enough space for cooktop and sink in
        # case of a I-form
        if self.depth > self.width / 3:
            self.depth = self.width / 3
        self.__origin = origin
        self.rotation = rotation
        self.shape = shape
        if origin is None:
            self.__origin = np.array([0, 0])

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        self.__origin = np.array(origin)

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        rot = np.radians(self.rotation)

        # calculate offsets
        # cook top on the left of the L- or U-kitchen, in the middle (rotation = 0)
        # sink in the middle of the bottom (rotation = 0)
        side_offset = [-np.sin(rot) * self.depth, np.cos(rot) * self.depth]
        cook_offset_left = [-np.sin(rot) * 0.5 * self.length, np.cos(rot) * 0.5 * self.length]
        cook_offset_right = [cook_offset_left[0] - np.sin(rot) * self.depth,
                             cook_offset_left[1] + np.cos(rot) * self.depth]
        cook_offset_top = [np.cos(rot) * self.depth, np.sin(rot) * self.depth]
        sink_offset = [np.cos(rot) * 0.5 * (self.width - 1.4 * self.depth) - 0.2 * self.depth * np.sin(rot),
                       np.sin(rot) * 0.5 * (self.width - 1.4 * self.depth) + 0.2 * self.depth * np.cos(rot)]

        # same distance between walls, cook top and sink
        if self.shape == 'I':
            # cook top width = self.depth, sink width = 1.4 * self.depth
            space_in_between = (self.width - 2.4 * self.depth) / 3
            cook_offset_left = [np.cos(rot) * space_in_between, np.sin(rot) * space_in_between]
            cook_offset_right = [cook_offset_left[0] + np.cos(rot) * self.depth,
                                 cook_offset_left[1] + np.sin(rot) * self.depth]
            cook_offset_top = [-np.sin(rot) * self.depth, np.cos(rot) * self.depth]
            sink_offset = [cook_offset_right[0] + cook_offset_left[0] - 0.2 * self.depth * np.sin(rot),
                           cook_offset_right[1] + cook_offset_left[1] + 0.2 * self.depth * np.cos(rot)]

        # draw outline
        if self.shape == 'I':
            ax.add_patch(Rectangle(abs_origin, self.width, self.depth, self.rotation, fill=False, facecolor="gray",
                                   linewidth=1.2, alpha=0.4, edgecolor="gray"))
        path = []
        path.append([abs_origin[0] - np.sin(rot) * self.length, abs_origin[1] + np.cos(rot) * self.length])
        path.append([path[0][0] + np.cos(rot) * self.depth, path[0][1] + np.sin(rot) * self.depth])
        path.append([path[1][0] + np.sin(rot) * (self.length - self.depth),
                     path[1][1] - np.cos(rot) * (self.length - self.depth)])
        if self.shape == 'L':
            path.append([path[2][0] + np.cos(rot) * (self.width - self.depth),
                         path[2][1] + np.sin(rot) * (self.width - self.depth)])
            ax.add_patch(PathPatch(Path(path), fill=False, linewidth=1.2, edgecolor='gray', alpha=0.4))
        if self.shape == 'U':
            path.append([path[2][0] + np.cos(rot) * (self.width - 2 * self.depth),
                         path[2][1] + np.sin(rot) * (self.width - 2 * self.depth)])
            path.append([path[3][0] - np.sin(rot) * (self.length - self.depth),
                         path[3][1] + np.cos(rot) * (self.length - self.depth)])
            path.append([path[4][0] + np.cos(rot) * self.depth, path[4][1] + np.sin(rot) * self.depth])
            ax.add_patch(PathPatch(Path(path), fill=False, linewidth=1.2, edgecolor='gray', alpha=0.4))

        # cook top
        # separators on the side
        ax.add_patch(PathPatch(Path([abs_origin + cook_offset_left,
                                     [abs_origin[0] + cook_offset_left[0] + cook_offset_top[0],
                                      abs_origin[1] + cook_offset_left[1] + cook_offset_top[1]]]), fill=False,
                               linewidth=1.2, edgecolor='gray', alpha=0.4))

        ax.add_patch(PathPatch(Path([abs_origin + cook_offset_right,
                                     [abs_origin[0] + cook_offset_right[0] + cook_offset_top[0],
                                      abs_origin[1] + cook_offset_right[1] + cook_offset_top[1]]]), fill=False,
                               linewidth=1.2, edgecolor='gray', alpha=0.4))
        # cooking plates
        radius = self.depth / 4 * 0.7
        plate_offset = [(np.cos(rot) - np.sin(rot)) * self.depth / 4, (np.cos(rot) + np.sin(rot)) * self.depth / 4]
        ax.add_patch(Circle(abs_origin + cook_offset_left + plate_offset, radius, fill=False, linewidth=1.2,
                            edgecolor='gray', alpha=0.4))
        plate_offset = [3 * plate_offset[0], 3 * plate_offset[1]]
        ax.add_patch(Circle(abs_origin + cook_offset_left + plate_offset, radius, fill=False, linewidth=1.2,
                            edgecolor='gray', alpha=0.4))
        plate_offset = [plate_offset[0] - np.cos(rot) * self.depth / 2, plate_offset[1] - np.sin(rot) * self.depth / 2]
        ax.add_patch(Circle(abs_origin + cook_offset_left + plate_offset, radius, fill=False, linewidth=1.2,
                            edgecolor='gray', alpha=0.4))
        plate_offset = [(3 * np.cos(rot) - np.sin(rot)) * self.depth / 4,
                        (np.cos(rot) + 3 * np.sin(rot)) * self.depth / 4]
        ax.add_patch(Circle(abs_origin + cook_offset_left + plate_offset, radius, fill=False, linewidth=1.2,
                            edgecolor='gray', alpha=0.4))

        # sink
        # outer metal part
        height, width = self.depth * 0.6, self.depth * 1.4
        sink_rot_offset = [0, 0, height, width, 0]
        if self.rotation == 90 or self.rotation == 270:
            height, width = width, height
        ax.add_patch(FancyBboxPatch([abs_origin[0] + sink_offset[0] - sink_rot_offset[int(self.rotation / 90 + 1)],
                                     abs_origin[1] + sink_offset[1] - sink_rot_offset[int(self.rotation / 90)]],
                                    width, height, boxstyle='round, pad=0, rounding_size=0.1',
                                    mutation_scale=self.scale, fill=False, linewidth=1.2, edgecolor='gray', alpha=0.4))

        # inner sink
        inner_sink_offset = [sink_offset[0] + 0.075 * self.depth * (-np.sin(rot) + np.cos(rot)),
                             sink_offset[1] + 0.075 * self.depth * (np.cos(rot) + np.sin(rot))]
        height, width = self.depth * 0.45, self.depth * 0.55
        sink_rot_offset = [0, 0, height, width, 0]
        if self.rotation == 90 or self.rotation == 270:
            height, width = width, height
        ax.add_patch(FancyBboxPatch(
            [abs_origin[0] + inner_sink_offset[0] - sink_rot_offset[int(self.rotation / 90 + 1)],
             abs_origin[1] + inner_sink_offset[1] - sink_rot_offset[int(self.rotation / 90)]], width, height,
            boxstyle='round, pad=0, rounding_size=0.1', mutation_scale=self.scale, fill=False, linewidth=1.2,
            edgecolor='gray', alpha=0.4))

        # draining board
        inner_sink_offset = [sink_offset[0] + self.depth * (-0.075 * np.sin(rot) + 0.775 * np.cos(rot)),
                             sink_offset[1] + self.depth * (0.075 * np.cos(rot) + 0.775 * np.sin(rot))]
        height, width = self.depth * 0.45, self.depth * 0.55
        sink_rot_offset = [0, 0, height, width, 0]
        if self.rotation == 90 or self.rotation == 270:
            height, width = width, height
        ax.add_patch(FancyBboxPatch(
            [abs_origin[0] + inner_sink_offset[0] - sink_rot_offset[int(self.rotation / 90 + 1)],
             abs_origin[1] + inner_sink_offset[1] - sink_rot_offset[int(self.rotation / 90)]], width, height,
            boxstyle='round, pad=0, rounding_size=0.1', mutation_scale=self.scale, fill=False, linewidth=1.2,
            edgecolor='gray', alpha=0.4))
        # pattern
        pattern_offset = [inner_sink_offset[0] + 0.05 * self.depth * (np.sin(rot) + np.cos(rot)),
                          inner_sink_offset[1] - 0.05 * self.depth * (np.cos(rot) - np.sin(rot))]
        for i in range(4):
            pattern_width = 0.45 * self.depth
            pattern_height = 0.05 * self.depth
            sec_offset = [0, 0]
            if i == 0 or i == 3:
                pattern_width = 0.35 * self.depth
                sec_offset = [0.05 * self.depth * np.cos(rot), 0.05 * self.depth * np.sin(rot)]
            pattern_offset[0] -= 0.1 * self.depth * np.sin(rot)
            pattern_offset[1] += 0.1 * self.depth * np.cos(rot)
            ax.add_patch(
                Rectangle(abs_origin + pattern_offset + sec_offset, pattern_width, pattern_height, self.rotation,
                          fill=False, facecolor="gray", linewidth=0.5, alpha=0.4, edgecolor="gray"))
