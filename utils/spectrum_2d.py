# spectrum_analyzer/spectral.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftfreq, fftshift


class SpectrumAnalyzer:
    """
    2D spectral analysis of a scalar ocean tracer field.

    Parameters
    ----------
    field             : 2D numpy array (lat, lon), may contain NaNs
    grid_spacing_deg  : spatial resolution in degrees (default 0.25°)
    label             : tracer name used in plot titles/labels

    Usage
    -----
        sa = SpectrumAnalyzer(field, label="O₂ (mmol/m³)")
        sa.compute()
        sa.plot_2d_spectrum(save_path="spectrum_2d.png")
        sa.plot_radial_spectrum(save_path="spectrum_radial.png")
    """

    KM_PER_DEG = 111.0

    def __init__(
        self,
        field: np.ndarray,
        grid_spacing_deg: float = 0.25,
        label: str = "Tracer",
    ) -> None:
        self.field            = field
        self.grid_spacing_deg = grid_spacing_deg
        self.label            = label

        # Populated by compute()
        self.power_2d = None
        self.kx       = None
        self.ky       = None

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _prepare(self) -> np.ndarray:
        """Fill NaNs with field mean and remove the DC component."""
        filled = np.where(np.isnan(self.field), np.nanmean(self.field), self.field)
        return filled - filled.mean()

    def _check_computed(self) -> None:
        if self.power_2d is None:
            raise RuntimeError("Call .compute() before plotting.")

    @staticmethod
    def _maybe_save(save_path) -> None:
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved → {save_path}")

    def _radial_average(self, n_bins: int):
        """Bin power_2d into radial wavenumber shells."""
        KX, KY   = np.meshgrid(self.kx, self.ky)
        K_radial = np.sqrt(KX**2 + KY**2)

        k_max   = K_radial.max()
        edges   = np.linspace(0, k_max, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        radial_power = np.array([
            self.power_2d[(K_radial >= edges[i]) & (K_radial < edges[i + 1])].mean()
            for i in range(n_bins)
        ])

        spacing_km  = self.grid_spacing_deg * self.KM_PER_DEG
        wavelengths = np.where(centers > 0, spacing_km / centers, np.nan)

        return centers, radial_power, wavelengths

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute(self):
        """
        Compute the 2D power spectrum.
        Must be called before any plot method.
        Returns self so calls can be chained.
        """
        prepped       = self._prepare()
        F_shifted     = fftshift(fft2(prepped))
        self.power_2d = np.abs(F_shifted) ** 2

        ny, nx  = self.power_2d.shape
        self.kx = fftshift(fftfreq(nx))
        self.ky = fftshift(fftfreq(ny))

        return self

    def plot_2d_spectrum(self, save_path=None) -> None:
        """Plot the full 2D power spectrum with wavelength axes."""
        self._check_computed()

        spacing_km = self.grid_spacing_deg * self.KM_PER_DEG

        kx_pos = self.kx[self.kx > 0]
        ky_pos = self.ky[self.ky > 0]
        wl_x_min = spacing_km / kx_pos.max()
        wl_x_max = spacing_km / kx_pos.min()
        wl_y_min = spacing_km / ky_pos.max()
        wl_y_max = spacing_km / ky_pos.min()

        fig, ax = plt.subplots(figsize=(8, 6))
        pcm = ax.imshow(
            np.log10(self.power_2d + 1e-10),
            origin="lower",
            extent=[wl_x_min, wl_x_max, wl_y_min, wl_y_max],
            cmap="plasma",
            aspect="auto",
        )
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label("log₁₀(Power)")
        ax.set_xlabel("Zonal wavelength (km)")
        ax.set_ylabel("Meridional wavelength (km)")
        ax.set_title(f"2D Power Spectrum — {self.label}")
        plt.tight_layout()
        self._maybe_save(save_path)
        plt.show()

    def plot_radial_spectrum(self, n_bins: int = 50, save_path=None) -> None:
        """Plot the azimuthally-averaged (radial) power spectrum."""
        self._check_computed()

        k_centers, radial_power, wavelengths = self._radial_average(n_bins)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(wavelengths, radial_power, color="darkblue", lw=2)
        ax.set_xlabel("Wavelength (km)")
        ax.set_ylabel("Power")
        ax.set_title(f"Radial Power Spectrum — {self.label}")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        self._maybe_save(save_path)
        plt.show()
