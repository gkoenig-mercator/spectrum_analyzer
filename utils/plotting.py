import numpy as np
import matplotlib.pyplot as plt
from utils.spectral import SpectrumData, KM_PER_DEG


def _maybe_save(save_path) -> None:
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")


def plot_2d_spectrum(spectrum: SpectrumData, units: str = "", 
                     save_path=None) -> None:
    if units:
        power_label = f"Power (({units})² · km²)"
    else:
        power_label = "Power (km²)"

    spacing_km = spectrum.grid_spacing_deg * KM_PER_DEG

    kx_pos   = spectrum.kx[spectrum.kx > 0]
    ky_pos   = spectrum.ky[spectrum.ky > 0]
    wl_x_min = spacing_km / kx_pos.max()
    wl_x_max = spacing_km / kx_pos.min()
    wl_y_min = spacing_km / ky_pos.max()
    wl_y_max = spacing_km / ky_pos.min()

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.imshow(
        np.log10(spectrum.power_2d + 1e-10),
        origin="lower",
        extent=[wl_x_min, wl_x_max, wl_y_min, wl_y_max],
        cmap="plasma",
        aspect="auto",
    )
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(f"log₁₀({power_label})")
    ax.set_xlabel("Zonal wavelength (km)")
    ax.set_ylabel("Meridional wavelength (km)")
    ax.set_title(f"2D Power Spectrum — {spectrum.label}")
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()


def plot_radial_spectrum(spectrum: SpectrumData, units: str = "", save_path=None) -> None:
    if units:
        power_label = f"Power (({units})² · km²)"
    else:
        power_label = "Power (km²)"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(spectrum.wavelengths, spectrum.radial_power, color="darkblue", lw=2)
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel(power_label)
    ax.set_title(f"Radial Power Spectrum — {spectrum.label}")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.invert_xaxis()
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()

