import json
import numpy as np
from scipy.fft import fft2, fftfreq, fftshift
from dataclasses import dataclass


KM_PER_DEG = 111.0


@dataclass
class SpectrumData:
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
        # Fill NaNs and remove DC component
        filled  = np.where(np.isnan(field), np.nanmean(field), field)
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

    def export(self, path: str) -> None:
        """
        Export the radial spectrum to JSON.
        The 2D power array is intentionally left out to keep the file small.
        """
        data = {
            "label"            : self.label,
            "grid_spacing_deg" : self.grid_spacing_deg,
            "k_centers"        : self.k_centers.tolist(),
            "radial_power"     : self.radial_power.tolist(),
            "wavelengths"      : self.wavelengths.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved → {path}")
