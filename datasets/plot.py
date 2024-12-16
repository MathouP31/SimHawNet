from multiprocessing import connection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


def draw_arrow(from_xy, to_xy, ax=None, alpha=0.1, **kwargs):
    if ax is None:
        ax = plt.gca()
    style = "Simple, tail_width=0.1, head_width=0.1, head_length=0.4"
    arrow = patches.FancyArrowPatch(from_xy,
                                    to_xy,
                                    connectionstyle="arc3,rad=.15",
                                    arrowstyle=style,
                                    antialiased=True,
                                    **kwargs)
    ax.add_patch(arrow)


def plot_temporal_graph(dataset,
                        ax,
                        cmap=plt.cm.viridis,
                        connection_hparams={
                            "alpha": 0.3,
                            "color": "k",
                        },
                        lines_hparams={
                            "alpha": 0.1,
                            "linestyle": 'dashed',
                        },
                        *args,
                        **kwargs):
    nodes = dataset.nodes

    tmin = min(dataset.t)
    tmax = max(dataset.t)
    norm = plt.Normalize(min(nodes), max(nodes))
    colors = cmap(norm(nodes))
    ax.hlines(nodes, tmin, tmax, colors=colors, **lines_hparams)
    for u, v, t in zip(dataset.src, dataset.dst, dataset.t):
        ax.scatter(t, u, color=cmap(norm(u)), *args, **kwargs)
        ax.scatter(t, v, color=cmap(norm(v)), *args, **kwargs)
        draw_arrow([t, u], [t, v], ax=ax, **connection_hparams)
    return ax
