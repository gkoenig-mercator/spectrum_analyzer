from utils.config import TRACER_CONFIG
import numpy as np
import copernicusmarine

def load_tracer_data(args) -> np.ndarray:
    cfg = TRACER_CONFIG[args.tracer]

    ds = copernicusmarine.open_dataset(
        dataset_id=cfg["dataset_id"],
        variables=[args.tracer],
        minimum_longitude=args.lon_min,
        maximum_longitude=args.lon_max,
        minimum_latitude=args.lat_min,
        maximum_latitude=args.lat_max,
        start_datetime=args.start,
        end_datetime=args.end,
        minimum_depth=args.depth,
        maximum_depth=args.depth,
    )

    field = (
        ds[args.tracer]
        .sel(depth=args.depth, method="nearest")
        .sel(time=args.start, method="nearest")
        .values
    )

    print(f"Loaded {args.tracer} at depth {args.depth}m, time={ds.time.values[0]}")
    return field
