from matplotlib.patches import Rectangle

from ._table import BaseTable


class DiningTable(BaseTable):
    def __init__(self, sits=8, rotation=0, height=1, width=2, scale=1, origin=None, **kwargs):
        super().__init__(height=height, width=width, origin=origin, sits=sits, rotation=rotation, scale=scale, **kwargs)

    def draw(self, ax, bbox=True, origin=(0, 0)):
        abs_origin = self.origin + origin
        ax.add_patch(Rectangle(abs_origin, self.width, self.height,
                               fill=True, facecolor="gray", linewidth=1.2,
                               alpha=0.4,
                               edgecolor='gray'))
