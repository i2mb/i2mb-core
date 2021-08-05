from matplotlib.patches import Rectangle, Circle, PathPatch
from i2mb.worlds import CompositeWorld
from i2mb.worlds.furniture.busCabin import BusCabin
from matplotlib.path import Path


class BusStation(CompositeWorld):

    def __init__(self, _rotation=0, cabinDims=(2, 1), **kwargs):
        super().__init__(**kwargs)
        self.cabin = BusCabin(origin=(
            self.origin[0] + (self.dims[0] - cabinDims[0]) / 2, self.origin[1] + self.dims[1] * 0.92 - cabinDims[1]),
            length=cabinDims[1], width=cabinDims[0])

    def _draw_world(self, ax=None, bbox=False):
        ax.add_patch(Rectangle(self.origin, *self.dims, fill=False, linewidth=1.2, edgecolor='gray'))
        self.cabin.draw(ax, bbox)
        radius = 0.1 * min(self.dims)
        # bus stop sign
        circle_origin = self.origin + (1.5 * radius)
        line_width = 6*radius
        ax.add_patch(
            Circle(circle_origin, radius=radius, fill=True, linewidth = 0.7*line_width, edgecolor='green', facecolor='yellow'))
        path_origin = circle_origin - 0.6 * radius + (0.2 * radius, 0)
        ax.add_patch(PathPatch(Path([path_origin, [path_origin[0], path_origin[1] + 1.2 * radius]]),
                               fill=False, linewidth = line_width, edgecolor='green'))
        path_origin[0] += 0.8*radius
        ax.add_patch(PathPatch(Path([path_origin, [path_origin[0], path_origin[1] + 1.2 * radius]]),
                               fill=False, linewidth = line_width, edgecolor='green'))
        path_origin[1] += 0.6*radius
        ax.add_patch(PathPatch(Path([path_origin, [path_origin[0] - 0.8*radius, path_origin[1]]]),
                      fill=False, linewidth = line_width, edgecolor='green'))

