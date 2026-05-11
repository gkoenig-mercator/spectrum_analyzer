import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from scipy.stats import gaussian_kde


def plot_distribution(
    data: np.ndarray,
    label: str = "Tracer",
    save_path: str = None,
) -> None:
    flat = data[~np.isnan(data)].flatten()
    mean_val   = np.mean(flat)
    median_val = np.median(flat)

    kde     = gaussian_kde(flat)
    x_range = np.linspace(flat.min(), flat.max(), 300)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(flat, bins=40, density=True,
            color="steelblue", alpha=0.6, label="Histogram")
    ax.plot(x_range, kde(x_range),
            color="darkblue", lw=2, label="KDE")
    ax.axvline(mean_val,   color="red",    linestyle="--",
               label=f"Mean: {mean_val:.3f}")
    ax.axvline(median_val, color="orange", linestyle="--",
               label=f"Median: {median_val:.3f}")

    ax.set_xlabel(label)                          # variable, not string
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {label}")      # also use it in title
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
