import matplotlib.pyplot as plt
from matplotlib import animation
from IPython import display


def cluster_animation(
    X, cluster_id_history, cluster_centroids_history, wcss_history, interval=20
):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    cmap = dict(enumerate(['blue','yellow','green','red','purple'] * 10))

    iterations = min(len(cluster_id_history), len(cluster_centroids_history), len(wcss_history))

    scat1 = ax[0].scatter(X[:, 0], X[:, 1], color=[cmap[i] for i in cluster_id_history[0]], label="observations")
    scat2 = ax[0].scatter(
        x=cluster_centroids_history[0][:, 0],
        y=cluster_centroids_history[0][:, 1],
        color="black",
        marker="o",
        s=200,
        edgecolors="yellow",
        label="centroids"
    )
    ax[0].set_title("clusters")
    ax[0].set_title("wcss")

    wcss_line, = ax[1].plot(wcss_history, "o-", label="score")

    ax[0].legend()
    ax[1].legend()

    fig.tight_layout()

    def draw_frame(n):
        wcss_line.set_data(list(range(n)), wcss_history[:n])
        scat1.set_color([cmap[i] for i in cluster_id_history[n]])
        scat2.set_offsets(
            cluster_centroids_history[n]
        )
        return (wcss_line, scat1, scat2, )

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=iterations, interval=interval, blit=False
    )
    return display.HTML(anim.to_html5_video())
