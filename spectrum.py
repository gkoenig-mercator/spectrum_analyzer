import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.stats import gaussian_kde
import copernicusmarine
import xarray as xr


# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────

def load_oxygen_data() -> xr.DataArray:
    """
    Load dissolved oxygen data from Copernicus Marine Service.
    Returns a 2D (lat, lon) DataArray squeezed from time and depth.
    """
    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
        variables=["o2"],
        minimum_longitude=-152,
        maximum_longitude=-80,
        minimum_latitude=-57,
        maximum_latitude=-25,
        start_datetime="2025-05-07T00:00:00",
        end_datetime="2025-05-07T00:00:00",
        maximum_depth=0.5,
    )
    o2 = ds["o2"].squeeze()
    print(f"Shape     : {o2.shape}")
    print(f"Range     : {float(o2.min()):.4f} – {float(o2.max()):.4f} mmol/m³")
    print(f"NaN count : {int(np.isnan(o2.values).sum())}")
    return o2


# ─────────────────────────────────────────
# DISTRIBUTION
# ─────────────────────────────────────────

def plot_distribution(o2: xr.DataArray, save_path: str = None) -> None:
    """
    Plot histogram and KDE of oxygen values with mean and median markers.

    Parameters
    ----------
    o2        : 2D oxygen DataArray (lat, lon)
    save_path : optional file path to save the figure (e.g. 'o2_dist.png')
    """
    o2_flat = o2.values[~np.isnan(o2.values)].flatten()
    mean_val   = np.mean(o2_flat)
    median_val = np.median(o2_flat)

    kde     = gaussian_kde(o2_flat)
    x_range = np.linspace(o2_flat.min(), o2_flat.max(), 300)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(o2_flat, bins=40, density=True,
            color="steelblue", alpha=0.6, label="Histogram")
    ax.plot(x_range, kde(x_range),
            color="darkblue", lw=2, label="KDE")
    ax.axvline(mean_val,   color="red",    linestyle="--",
               label=f"Mean: {mean_val:.3f}")
    ax.axvline(median_val, color="orange", linestyle="--",
               label=f"Median: {median_val:.3f}")

    ax.set_xlabel("O₂ (mmol/m³)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Dissolved Oxygen Values")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────
# 2D POWER SPECTRUM
# ─────────────────────────────────────────

def _prepare_field_for_fft(o2_vals: np.ndarray) -> np.ndarray:
    """
    Fill NaNs with field mean and remove the overall mean
    to avoid a DC spike dominating the spectrum.
    Internal helper — not intended for direct use.
    """
    filled    = np.where(np.isnan(o2_vals), np.nanmean(o2_vals), o2_vals)
    detrended = filled - filled.mean()
    return detrended


def compute_2d_power_spectrum(o2: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 2D power spectrum of the oxygen field.

    Returns
    -------
    power_2d : 2D array of power values (shifted, zero-frequency centred)
    kx       : 1D array of zonal wavenumbers      (cycles / grid point)
    ky       : 1D array of meridional wavenumbers  (cycles / grid point)
    """
    o2_vals   = o2.values
    prepped   = _prepare_field_for_fft(o2_vals)
    F_shifted = fft.fftshift(fft.fft2(prepped))
    power_2d  = np.abs(F_shifted) ** 2

    ny, nx = power_2d.shape
    kx = fft.fftshift(fft.fftfreq(nx))
    ky = fft.fftshift(fft.fftfreq(ny))

    return power_2d, kx, ky

def plot_2d_spectrum(
    power_2d: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    grid_spacing_deg: float = 0.25,
    save_path: str = None,
) -> None:
    km_per_deg = 111.0

    # Max wavelength (min nonzero wavenumber), min wavelength (max wavenumber)
    kx_pos = kx[kx > 0]
    ky_pos = ky[ky > 0]
    wl_x_max = grid_spacing_deg * km_per_deg / kx_pos.min()
    wl_x_min = grid_spacing_deg * km_per_deg / kx_pos.max()
    wl_y_max = grid_spacing_deg * km_per_deg / ky_pos.min()
    wl_y_min = grid_spacing_deg * km_per_deg / ky_pos.max()

    fig, ax = plt.subplots(figsize=(8, 6))

    pcm = ax.imshow(
        np.log10(power_2d + 1e-10),
        origin="lower",
        extent=[wl_x_min, wl_x_max, wl_y_min, wl_y_max],
        cmap="plasma",
        aspect="auto",
    )
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label("log₁₀(Power)  [(mmol/m³)² · km²]")

    ax.set_xlabel("Zonal wavelength (km)")
    ax.set_ylabel("Meridional wavelength (km)")
    ax.set_title("2D Power Spectrum of Dissolved Oxygen")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────
# RADIAL SPECTRUM
# ─────────────────────────────────────────

def compute_radial_spectrum(
    power_2d: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    grid_spacing_deg: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Azimuthally average the 2D power spectrum into a 1D radial spectrum.

    Parameters
    ----------
    power_2d         : 2D power spectrum from compute_2d_power_spectrum()
    kx, ky           : wavenumber arrays from compute_2d_power_spectrum()
    grid_spacing_deg : spatial resolution of the data in degrees (default 0.25)

    Returns
    -------
    k_centers      : radial wavenumber bin centres (cycles / grid point)
    radial_power   : mean power in each bin
    wavelength_deg : spatial wavelength in degrees (= grid_spacing / k)
    """
    KX, KY = np.meshgrid(kx, ky)
    K      = np.sqrt(KX**2 + KY**2)

    ny, nx     = power_2d.shape
    k_bins     = np.linspace(0, K.max(), min(nx, ny) // 2)
    k_centers  = 0.5 * (k_bins[:-1] + k_bins[1:])
    radial_power = np.array([
        power_2d[(K >= k0) & (K < k1)].mean()
        for k0, k1 in zip(k_bins[:-1], k_bins[1:])
        if ((K >= k0) & (K < k1)).sum() > 0
    ])

    # Trim k_centers to match radial_power length (empty bins are dropped)
    k_centers     = k_centers[:len(radial_power)]
    wavelength_deg = np.where(k_centers > 0, grid_spacing_deg / k_centers, np.nan)

    return k_centers, radial_power, wavelength_deg


def plot_radial_spectrum(
    k_centers: np.ndarray,
    radial_power: np.ndarray,
    wavelength_deg: np.ndarray,
    grid_spacing_deg: float = 0.25,
    save_path: str = None,
) -> None:
    """
    Plot the radially averaged power spectrum with a regression line and slope.

    Parameters
    ----------
    k_centers        : radial wavenumber bin centres from compute_radial_spectrum()
    radial_power     : mean power per bin from compute_radial_spectrum()
    wavelength_deg   : wavelength in degrees from compute_radial_spectrum()
    grid_spacing_deg : grid resolution in degrees (default 0.25°)
    save_path        : optional file path to save the figure
    """
    km_per_deg   = 111.0
    wavelength_km = wavelength_deg * km_per_deg  # convert to km

    valid = ~np.isnan(wavelength_km) & (radial_power > 0)

    # ── Regression in log-log space ──────────────────────────────────────────
    log_k     = np.log10(k_centers[valid])
    log_power = np.log10(radial_power[valid])
    slope, intercept = np.polyfit(log_k, log_power, deg=1)
    regression_line  = 10 ** (slope * log_k + intercept)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel : power vs wavenumber ────────────────────────────────────
    axes[0].loglog(k_centers[valid], radial_power[valid],
                   color="steelblue", lw=2, label="Radial power")
    axes[0].loglog(k_centers[valid], regression_line,
                   color="red", lw=1.5, linestyle="--",
                   label=f"Slope: {slope:.2f}")
    axes[0].set_xlabel("Radial wavenumber (cycles / grid point)")
    axes[0].set_ylabel("Power  [(mmol/m³)²]")
    axes[0].set_title("Radially Averaged Power Spectrum")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)

    # ── Right panel : power vs wavelength in km ──────────────────────────────
    axes[1].loglog(wavelength_km[valid], radial_power[valid],
                   color="darkorange", lw=2, label="Radial power")
    axes[1].loglog(wavelength_km[valid], regression_line,
                   color="red", lw=1.5, linestyle="--",
                   label=f"Slope: {slope:.2f}")
    axes[1].set_xlabel("Spatial wavelength (km)")
    axes[1].set_ylabel("Power  [(mmol/m³)²]")
    axes[1].set_title("Power Spectrum vs Spatial Scale")
    axes[1].invert_xaxis()  # large scales on the left
    axes[1].legend()
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    o2 = load_oxygen_data()

    plot_distribution(o2, save_path="o2_distribution.png")

    power_2d, kx, ky = compute_2d_power_spectrum(o2)
    plot_2d_spectrum(power_2d, kx, ky, save_path="o2_2d_spectrum.png")

    k_centers, radial_power, wavelength_deg = compute_radial_spectrum(power_2d, kx, ky)
    plot_radial_spectrum(k_centers, radial_power, wavelength_deg,
                         save_path="o2_radial_spectrum.png")
