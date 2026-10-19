"""
Shared configuration for the whole workflow.

Every script imports from here so training, explaining and serving all
point at the same MLflow store, experiment, model name and log file.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

import mlflow

BASE = os.path.dirname(os.path.abspath(__file__))

# ---- MLflow ----
TRACKING_URI = "sqlite:///" + os.path.join(BASE, "mlflow.db")  # registry needs a DB store
EXPERIMENT = "breast-cancer-automl"
MODEL_NAME = "breast-cancer-classifier"
MODEL_ALIAS = "champion"          # the service always loads the 'champion' version

# ---- files produced by training, used by the service for drift checks ----
ARTIFACT_DIR = os.path.join(BASE, "artifacts")
TRAIN_REFERENCE = os.path.join(ARTIFACT_DIR, "train_reference.csv")

# ---- logging ----
LOG_DIR = os.path.join(BASE, "logs")
LOG_FILE = os.path.join(LOG_DIR, "service.log")


def setup_mlflow():
    """Point MLflow at the local store and select the experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


def get_logger(name: str = "ml-service") -> logging.Logger:
    """
    Structured logger: console + rotating file, timestamp | level | name | message.
    Level comes from the LOG_LEVEL environment variable (default INFO).
    """
    logger = logging.getLogger(name)
    if logger.handlers:            # avoid duplicate handlers on re-import
        return logger
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)

    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    fileh = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3,
                                encoding="utf-8")
    fileh.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(fileh)
    logger.propagate = False
    return logger
