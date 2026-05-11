from utils.config import TRACER_CONFIG
import numpy as np
import copernicusmarine

def load_tracer_data(args) -> np.ndarray:
    cfg = TRACER_CONFIG[args.tracer]

    ds = copernicusmarine.open_dataset(
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
    print(ds)

    field = ds[args.tracer].isel(time=0, depth=0).values
    return field
