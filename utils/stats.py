import numpy as np
from scipy.stats import gaussian_kde
from dataclasses import dataclass, asdict
import json

@dataclass
class DistributionStats:
    flat:    np.ndarray
    mean:    float
    median:  float
    std:     float
    q25:     float
    q75:     float
    kde:     gaussian_kde
    x_range: np.ndarray

    @classmethod
    def from_data(cls, data: np.ndarray) -> "DistributionStats":
        flat = data[~np.isnan(data)].flatten()
        x_range = np.linspace(flat.min(), flat.max(), 300)
        return cls(
            flat    = flat,
            mean    = float(np.mean(flat)),
            median  = float(np.median(flat)),
            std     = float(np.std(flat)),
            q25     = float(np.percentile(flat, 25)),
            q75     = float(np.percentile(flat, 75)),
            kde     = gaussian_kde(flat),
            x_range = x_range,
        )

    def export(self, path: str) -> None:
        payload = {
            "mean":   self.mean,
            "median": self.median,
            "std":    self.std,
            "q25":    self.q25,
            "q75":    self.q75,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved → {path}")
