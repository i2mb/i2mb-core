import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def scale_bar(ax, reference_div=10, num_partitions=4, num_divs=3, position=None):
    """Draws a scale bar in ax. The reference_div is the size of the largest block in data units divided in
    num_partitions. Additional divs are added with double the partitions of the previous block.

    The scale bar is placed at ´position´ data coordinates. """

    if position is None:
        position = (0, 0)

    position = np.array(position)
    positions = np.repeat(position.reshape(-1, 2), num_divs, axis=0)

    positions[:, 0] = [position[0] + i * reference_div for i in range(num_divs)]
    # x, y = np.diff(ax.transData.transform([[0, 0.], [0, 1.]]), axis=0).ravel()
    x, y = ax.figure.transFigure.transform([0, 0.005])
    ax_x, ax_y = ax.transAxes.transform(position)
    bbox_anchor = ax.transAxes.inverted().transform([ax_x, ax_y - y])
    # bbox_anchor = ax.figure.transFigure.inverted().transform([ax_x, ax_y - y])
    ax.add_artist(AnchoredSizeBar(ax.transData, reference_div, f"{reference_div}m", "upper left", frameon=False,
                                  bbox_to_anchor=bbox_anchor,
                                  bbox_transform=ax.transAxes,
                                  prop={"size": 8}
                                  ))










