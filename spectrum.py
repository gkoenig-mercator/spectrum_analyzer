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
from utils.spectrum_analyzer import SpectrumAnalyzer


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

    sa = SpectrumAnalyzer(field, label=f"{cfg['long_name']} ({cfg['units']})")
    sa.compute()
    sa.plot_2d_spectrum(save_path=f"{args.save_dir_plots}/{t}_2d_spectrum.png")
    sa.plot_radial_spectrum(save_path=f"{args.save_dir_plots}/{t}_radial_spectrum.png")
