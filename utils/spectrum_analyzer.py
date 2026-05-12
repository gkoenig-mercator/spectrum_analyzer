# spectrum_analyzer/spectral.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftfreq, fftshift
from dataclasses import dataclass


KM_PER_DEG = 111.0


# ------------------------------------------------------------------ #
#  Dataclass                                                          #
# ------------------------------------------------------------------ #

@dataclass
class SpectrumData:
    """
    Holds the results of a 2D spectral analysis.

    Fields
    ------
    power_2d         : 2D power spectrum array
    kx, ky           : wavenumber arrays
    k_centers        : bin centers of the radial average
    radial_power     : azimuthally averaged power per bin
    wavelengths      : wavelength in km per bin
    label            : tracer name used in plot titles
    grid_spacing_deg : original grid spacing in degrees
    """
    power_2d         : np.ndarray
    kx               : np.ndarray
    ky               : np.ndarray
    k_centers        : np.ndarray
    radial_power     : np.ndarray
    wavelengths      : np.ndarray
    label            : str
    grid_spacing_deg : float

    @classmethod
    def from_field(
        cls,
        field            : np.ndarray,
        grid_spacing_deg : float = 0.25,
        label            : str   = "Tracer",
        n_bins           : int   = 50,
    ) -> "SpectrumData":
        """
        Compute the spectrum from a raw 2D field.

        Parameters
        ----------
        field            : 2D numpy array (lat, lon), may contain NaNs
        grid_spacing_deg : spatial resolution in degrees
        label            : tracer name for plot titles
        n_bins           : number of radial bins for the azimuthal average
        """
        # Fill NaNs and remove DC component
        filled = np.where(np.isnan(field), np.nanmean(field), field)
        prepped = filled - filled.mean()

        # 2D FFT
        F_shifted = fftshift(fft2(prepped))
        power_2d  = np.abs(F_shifted) ** 2

        ny, nx = power_2d.shape
        kx = fftshift(fftfreq(nx))
        ky = fftshift(fftfreq(ny))

        # Radial average
        KX, KY   = np.meshgrid(kx, ky)
        K_radial = np.sqrt(KX**2 + KY**2)
        k_max    = K_radial.max()
        edges    = np.linspace(0, k_max, n_bins + 1)
        centers  = 0.5 * (edges[:-1] + edges[1:])

        radial_power = np.array([
            power_2d[(K_radial >= edges[i]) & (K_radial < edges[i + 1])].mean()
            for i in range(n_bins)
        ])

        spacing_km  = grid_spacing_deg * KM_PER_DEG
        wavelengths = np.where(centers > 0, spacing_km / centers, np.nan)

        return cls(
            power_2d         = power_2d,
            kx               = kx,
            ky               = ky,
            k_centers        = centers,
            radial_power     = radial_power,
            wavelengths      = wavelengths,
            label            = label,
            grid_spacing_deg = grid_spacing_deg,
        )


# ------------------------------------------------------------------ #
#  Plot functions                                                      #
# ------------------------------------------------------------------ #

def _maybe_save(save_path) -> None:
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")


def plot_2d_spectrum(spectrum: SpectrumData, save_path=None) -> None:
    """Plot the full 2D power spectrum with wavelength axes."""
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
    cbar.set_label("log₁₀(Power)")
    ax.set_xlabel("Zonal wavelength (km)")
    ax.set_ylabel("Meridional wavelength (km)")
    ax.set_title(f"2D Power Spectrum — {spectrum.label}")
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()


def plot_radial_spectrum(spectrum: SpectrumData, save_path=None) -> None:
    """Plot the azimuthally-averaged (radial) power spectrum."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(spectrum.wavelengths, spectrum.radial_power, color="darkblue", lw=2)
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Power")
    ax.set_title(f"Radial Power Spectrum — {spectrum.label}")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()
