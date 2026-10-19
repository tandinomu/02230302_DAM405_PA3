"""
Monitoring signals and data-drift detection.

* stats      : request count, error count, last 100 latencies
* summary()  : one-line health summary (count, error rate, avg + p95 latency)
* DriftDetector : compares a window of live inputs with the training data
    - Kolmogorov-Smirnov test per feature (compares whole distributions)
    - relative change in the mean (easy to read, as in Practical 5)
  A feature is flagged when KS p-value < 0.01. A WARNING is logged when
  any feature drifts.
"""
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config import get_logger

logger = get_logger("ml-service")

stats = {"requests": 0, "errors": 0, "latencies": deque(maxlen=100)}


def record(latency_ms: float, ok: bool = True) -> None:
    stats["requests"] += 1
    if not ok:
        stats["errors"] += 1
    stats["latencies"].append(latency_ms)


def summary() -> dict:
    lat = list(stats["latencies"])
    avg = float(np.mean(lat)) if lat else 0.0
    p95 = float(np.percentile(lat, 95)) if lat else 0.0
    rate = stats["errors"] / stats["requests"] if stats["requests"] else 0.0
    logger.info(f"SUMMARY requests={stats['requests']} errors={stats['errors']} "
                f"error_rate={rate:.1%} avg_latency_ms={avg:.1f} p95_latency_ms={p95:.1f}")
    return {"requests": stats["requests"], "errors": stats["errors"],
            "error_rate": round(rate, 4), "avg_latency_ms": round(avg, 2),
            "p95_latency_ms": round(p95, 2)}


class DriftDetector:
    def __init__(self, reference: pd.DataFrame, window: int = 200,
                 min_samples: int = 30, p_threshold: float = 0.01):
        self.reference = reference
        self.live = deque(maxlen=window)        # most recent live inputs
        self.min_samples = min_samples
        self.p_threshold = p_threshold

    def add(self, row: list) -> None:
        self.live.append(row)

    def check(self) -> dict:
        """Run the KS test on every feature; log WARNING for drifted ones."""
        n = len(self.live)
        if n < self.min_samples:
            logger.info(f"drift check skipped: only {n} live samples "
                        f"(need {self.min_samples})")
            return {"status": "insufficient_data", "live_samples": n}

        live = pd.DataFrame(list(self.live), columns=self.reference.columns)
        drifted = []
        for col in self.reference.columns:
            stat, p = ks_2samp(self.reference[col], live[col])
            mean_change = abs(live[col].mean() - self.reference[col].mean()) / \
                (abs(self.reference[col].mean()) + 1e-9)
            if p < self.p_threshold:
                drifted.append({"feature": col, "ks_stat": round(float(stat), 3),
                                "p_value": float(f"{p:.2e}"),
                                "mean_change": round(float(mean_change), 3)})
                logger.warning(f"DRIFT feature='{col}' ks={stat:.3f} p={p:.2e} "
                               f"mean_change={mean_change:.0%}")

        share = len(drifted) / len(self.reference.columns)
        if drifted:
            logger.warning(f"DRIFT DETECTED in {len(drifted)}/"
                           f"{len(self.reference.columns)} features ({share:.0%}) "
                           f"over last {n} requests")
        else:
            logger.info(f"no drift: 0/{len(self.reference.columns)} features "
                        f"flagged over last {n} requests")
        return {"status": "drift" if drifted else "ok", "live_samples": n,
                "drifted_features": len(drifted), "drift_share": round(share, 3),
                "details": drifted}

    def reset(self) -> None:
        self.live.clear()
