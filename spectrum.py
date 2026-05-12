import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.stats import gaussian_kde
import xarray as xr
from utils.config import TRACER_CONFIG

from utils.cli import parse_args
from utils.data_import import load_tracer_data
from utils.stats import DistributionStats
from utils.distribution import plot_distribution
from utils.spectrum_analyzer import SpectrumData, plot_2d_spectrum, plot_radial_spectrum


if __name__ == "__main__":

    args = parse_args()
    cfg = TRACER_CONFIG[args.tracer]
    t = args.tracer  # short alias for filenames

    field = load_tracer_data(args)
    stats = DistributionStats.from_data(field)

    plot_distribution(
        stats,
        label=f"{cfg['long_name']} ({cfg['units']})",
        save_path=f"{args.save_dir_plots}/{t}_distribution.png",
    )
    stats.export(f"{args.save_dir_data}/{t}_stats.json")

    spectrum = SpectrumData.from_field(field, label=f"{cfg['long_name']} ({cfg['units']})")
    # Access fields directly
    print(spectrum.wavelengths)
    print(spectrum.radial_power)
    plot_2d_spectrum(spectrum, save_path=f"{args.save_dir_plots}/{t}_2d_spectrum.png")
    plot_radial_spectrum(spectrum, save_path=f"{args.save_dir_plots}/{t}_radial_spectrum.png")
