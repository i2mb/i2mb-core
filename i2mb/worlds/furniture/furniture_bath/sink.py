from matplotlib.patches import Rectangle

from i2mb.worlds.furniture.base_furniture import BaseFurniture


class Sink(BaseFurniture):
    def __init__(self,  rotation=0, origin=None, width=0.5, height=0.7, scale=1):
        super().__init__(height, width, origin, rotation, scale)
        self.use_position = [width, height/2]
        self.points.extend([self.use_position])

    def get_activity_position(self, origin=(0, 0), pos_id=None):
        return self.origin + self.use_position + origin

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = origin + self.origin

        ax.add_patch(Rectangle(abs_origin, self.width, self.height,
                               fill=True, facecolor="gray", linewidth=1.2,
                               alpha=0.4, edgecolor="gray"))

        ax.add_patch(Rectangle((abs_origin + self.dims*0.1), self.width * 0.8, self.height * 0.8,
                               fill=True, facecolor="white", linewidth=1.2, edgecolor=None))
