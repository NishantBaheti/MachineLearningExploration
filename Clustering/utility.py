import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation
import seaborn as sns
from IPython import display
import numpy as np


def cluster_animation(
    X, cluster_id_history, cluster_centroids_history, iterations, interval=20
):
    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(1, 1, 1)
    cmap = dict(enumerate(['blue','yellow','green','red','purple']))

    scat1 = ax.scatter(X[:, 0], X[:, 1], color=[cmap[i] for i in cluster_id_history[0]])
    scat2 = ax.scatter(
        x=cluster_centroids_history[0][:, 0],
        y=cluster_centroids_history[0][:, 1],
        color="black",
        marker="o",
        s=200,
        edgecolors="yellow",
    )

    fig.tight_layout()

    def draw_frame(n):

        scat1.set_color([cmap[i] for i in cluster_id_history[n]])
        scat2.set_offsets(
            cluster_centroids_history[n]
        )
        return (scat1, scat2)

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=iterations, interval=interval, blit=False
    )
    return display.HTML(anim.to_html5_video())
