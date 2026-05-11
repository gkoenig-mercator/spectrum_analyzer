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

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    args = parse_args()
    cfg = TRACER_CONFIG[args.tracer]

    ds = copernicusmarine.subset(
        dataset_id=cfg["dataset_id"],
        variables=[args.tracer],
        minimum_longitude=-60.830714,
        maximum_longitude=-41.984272,
        minimum_latitude=21.906553,
        maximum_latitude=35.717104,
        start_datetime=args.start,
        end_datetime=args.end,
        minimum_depth=args.depth,
        maximum_depth=args.depth,
    )

    # extract field — works for any tracer name now
    field = ds[args.tracer].isel(time=0, depth=0).values

    # pass tracer metadata to plot functions
    label = f"{cfg['long_name']} ({cfg['units']})"

    power_2d, kx, ky = compute_2d_spectrum(field)
    plot_2d_spectrum(power_2d, kx, ky,
                     save_path=f"{args.save_dir}/{args.tracer}_2d_spectrum.png")

    plot_distribution(field, label=label,
                      save_path=f"{args.save_dir}/{args.tracer}_distribution.png")

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
    sa.plot_2d_spectrum(save_path="o2_2d_spectrum.png")
    sa.plot_radial_spectrum(save_path="o2_radial_spectrum.png")
