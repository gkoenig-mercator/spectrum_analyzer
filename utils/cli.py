import argparse
from utils.config import TRACER_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot Fourier spectrum of ocean tracers."
    )
    parser.add_argument(
        "--tracer",
        type=str,
        default="o2",
        choices=list(TRACER_CONFIG.keys()),
        help=f"Tracer variable to analyse. Choices: {list(TRACER_CONFIG.keys())}",
    )
    parser.add_argument(
        "--start", type=str, default="2026-05-07T00:00:00",
        help="Start datetime (ISO format)"
    )
    parser.add_argument(
        "--end", type=str, default="2026-05-07T00:00:00",
        help="End datetime (ISO format)"
    )
    parser.add_argument(
        "--depth", type=float, default=2533.336181640625,
        help="Target depth in metres"
    )
    parser.add_argument(
        "--save-dir-plots", type=str, default="./plots",
        help="Directory to save output figures"
    )
    parser.add_argument(
        "--save-dir-data", type=str, default="./data",
        help="Directory to save output data"
    )

    parser.add_argument(
        "--lon-min", type=float, default=-60.
    )
    parser.add_argument(
        "--lon-max", type=float, default=-41.
    )
    parser.add_argument(
        "--lat-min", type=float, default=21.
    )
    parser.add_argument(
        "--lat-max", type=float, default=35.
    )
    return parser.parse_args()
