from matplotlib.patches import Rectangle


def draw_world(world, ax=None, padding=5):
    ax.set_axis_off()
    ax.set_aspect("equal")

    ax.add_patch(Rectangle(world.origin, *world.dims, fill=False, linewidth=1.2, edgecolor='gray'))

    for r in world.regions:
        ax.add_patch(Rectangle(r.origin, *r.dims, fill=False, linewidth=1.2))

    ax.set_xlim(world.origin[0] - padding, world.dims[0] + padding)
    ax.set_ylim(world.origin[1] - padding, world.dims[1] + padding)

    # Draw agents
    return ax.scatter(*world.get_absolute_positions().T, s=16)
