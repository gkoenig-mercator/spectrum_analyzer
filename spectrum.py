import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.stats import gaussian_kde
import xarray as xr
from utils.config import TRACER_CONFIG

from utils.cli import parse_args
from utils.data_import import load_tracer_data
from utils.distribution import plot_distribution
from utils.spectrum_analyzer import SpectrumAnalyzer


if __name__ == "__main__":

    args = parse_args()
    cfg = TRACER_CONFIG[args.tracer]
    t = args.tracer  # short alias for filenames

    field = load_tracer_data(args)

    plot_distribution(
        field,
        label=f"{cfg['long_name']} ({cfg['units']})",
        save_path=f"{args.save_dir}/{t}_distribution.png",
    )

    sa = SpectrumAnalyzer(field, label="O₂ (mmol/m³)")
    sa.compute()
    sa.plot_2d_spectrum(save_path=f"{args.save_dir}/{t}_2d_spectrum.png")
    sa.plot_radial_spectrum(save_path=f"{args.save_dir}/{t}_radial_spectrum.png")
