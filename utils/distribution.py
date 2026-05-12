import matplotlib.pyplot as plt
import numpy as np
from utils.stats import DistributionStats

def plot_distribution(
    stats: DistributionStats,
    label: str = "Tracer",
    save_path: str = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(stats.flat, bins=40, density=True,
            color="steelblue", alpha=0.6, label="Histogram")
    ax.plot(stats.x_range, stats.kde(stats.x_range),
            color="darkblue", lw=2, label="KDE")
    ax.axvline(stats.mean,   color="red",    linestyle="--",
               label=f"Mean: {stats.mean:.3f}")
    ax.axvline(stats.median, color="orange", linestyle="--",
               label=f"Median: {stats.median:.3f}")

    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {label}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
