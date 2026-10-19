"""
Serve the registered model with logging and monitoring.

Loads models:/breast-cancer-classifier@champion from the MLflow Model Registry.

Endpoints
  POST /predict   {"features": [30 numbers]}  or  {"features": {"mean radius": ...}}
  GET  /stats     monitoring summary (requests, error rate, latency)
  GET  /drift     run the drift check on recent inputs
  POST /drift/reset   clear the live window
  GET  /health    liveness check

Log levels used
  DEBUG    raw request payload (development only)
  INFO     prediction + latency, summaries, startup
  WARNING  suspicious-but-handled input, data drift
  ERROR    a request failed (bad input -> 400, server fault -> 500)
  CRITICAL the model cannot be loaded -> service exits

Run:  python app.py      (LOG_LEVEL=DEBUG python app.py for verbose logs)
"""
import sys
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from config import (MODEL_ALIAS, MODEL_NAME, TRAIN_REFERENCE, get_logger,
                    setup_mlflow)
from monitoring import DriftDetector, record, stats, summary

logger = get_logger("ml-service")
SUMMARY_EVERY = 25           # log a summary line every N requests
DRIFT_EVERY = 50             # run the drift check automatically every N requests

app = Flask(__name__)

# ---- start-up: load model and reference data ----
try:
    setup_mlflow()
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    reference = pd.read_csv(TRAIN_REFERENCE)
    FEATURES = list(reference.columns)
    logger.info(f"loaded {MODEL_NAME}@{MODEL_ALIAS} ({type(model).__name__}), "
                f"{len(FEATURES)} features")
except Exception as e:
    logger.critical(f"cannot load model or reference data: {e}. Run train.py first.")
    sys.exit(1)

drift = DriftDetector(reference)
# range of each feature in training; values far outside are suspicious
LOW, HIGH = reference.min(), reference.max()


def parse_features(payload):
    """Accept a list in training order or a {feature: value} dict."""
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError("body must be JSON with a 'features' field")
    f = payload["features"]
    if isinstance(f, dict):
        missing = [c for c in FEATURES if c not in f]
        if missing:
            raise ValueError(f"missing features: {missing[:3]}...")
        f = [f[c] for c in FEATURES]
    if not isinstance(f, list) or len(f) != len(FEATURES):
        raise ValueError(f"expected {len(FEATURES)} features, got "
                         f"{len(f) if isinstance(f, list) else type(f).__name__}")
    return [float(v) for v in f]          # raises ValueError for non-numbers


@app.post("/predict")
def predict():
    start = time.perf_counter()
    try:
        payload = request.get_json(force=True, silent=True)
        logger.debug(f"request payload: {payload}")
        row = parse_features(payload)

        # out-of-range values are allowed but worth a warning
        x = pd.DataFrame([row], columns=FEATURES)
        out = [c for c in FEATURES if x[c].iloc[0] < LOW[c] or x[c].iloc[0] > HIGH[c]]
        if out:
            logger.warning(f"input outside training range for {len(out)} feature(s): "
                           f"{out[:3]}")

        proba = float(model.predict_proba(x)[0, 1])
        label = int(proba >= 0.5)
        drift.add(row)

        ms = (time.perf_counter() - start) * 1000
        logger.info(f"prediction={label} ({'benign' if label else 'malignant'}) "
                    f"p_benign={proba:.3f} latency_ms={ms:.1f}")
        record(ms, ok=True)
        response = (jsonify({"prediction": label,
                             "label": "benign" if label else "malignant",
                             "probability_benign": round(proba, 4)}), 200)

    except (ValueError, TypeError) as e:
        ms = (time.perf_counter() - start) * 1000
        logger.error(f"bad request: {e} latency_ms={ms:.1f}")
        record(ms, ok=False)
        response = (jsonify({"error": str(e)}), 400)

    except Exception as e:
        ms = (time.perf_counter() - start) * 1000
        logger.exception(f"prediction failed (server error): {e}")
        record(ms, ok=False)
        response = (jsonify({"error": "internal server error"}), 500)

    # periodic monitoring
    if stats["requests"] % SUMMARY_EVERY == 0:
        summary()
    if stats["requests"] % DRIFT_EVERY == 0:
        drift.check()
    return response


@app.get("/stats")
def get_stats():
    return jsonify(summary())


@app.get("/drift")
def get_drift():
    return jsonify(drift.check())


@app.post("/drift/reset")
def reset_drift():
    drift.reset()
    logger.info("drift window reset")
    return jsonify({"status": "reset"})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": f"{MODEL_NAME}@{MODEL_ALIAS}"})


if __name__ == "__main__":
    logger.info("starting service on http://127.0.0.1:5001")
    # port 5001 so it can run next to the MLflow UI on 5000
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
