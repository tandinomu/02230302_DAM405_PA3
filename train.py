"""
Train & track.

1. Baseline : logistic regression with default settings (scaled inputs).
2. AutoML   : FLAML searches model types and hyperparameters within a time budget.
3. Both are logged to MLflow (params, metrics, model, plots).
4. The better model is registered in the Model Registry and given the
   alias 'champion', which the service loads.

Run:  python train.py            (about 1-2 minutes)
      python train.py --budget 30   (shorter AutoML search)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")              # save plots without opening a window
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from flaml import AutoML
from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (ARTIFACT_DIR, EXPERIMENT, MODEL_ALIAS, MODEL_NAME,
                    TRAIN_REFERENCE, get_logger, setup_mlflow)

logger = get_logger("train")
CLOUDPICKLE = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE  # works for tree models


def load_data():
    """Breast cancer dataset: 569 samples, 30 numeric features, binary target."""
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target          # 0 = malignant, 1 = benign
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def evaluate(model, X_test, y_test):
    """The metrics that matter for a medical classifier, not just accuracy."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def cv_roc_auc(model, X, y):
    """5-fold cross-validated ROC-AUC on the training data: a fairer comparison
    than the small 114-sample test set alone."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(clone(model), X, y, cv=cv, scoring="roc_auc").mean()


def log_confusion_matrix(model, X_test, y_test, name):
    """Save a confusion matrix plot and attach it to the current run."""
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["malignant", "benign"], cmap="Blues")
    disp.ax_.set_title(name)
    path = os.path.join(ARTIFACT_DIR, f"confusion_{name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    mlflow.log_artifact(path, artifact_path="plots")


def main(budget):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    logger.info(f"data loaded: train={X_train.shape} test={X_test.shape}")

    # Save the training features: the service compares live inputs to these (drift)
    X_train.to_csv(TRAIN_REFERENCE, index=False)

    setup_mlflow()
    results = {}

    # ---------- 1. Baseline ----------
    with mlflow.start_run(run_name="baseline_logreg") as run:
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        baseline.fit(X_train, y_train)
        metrics = evaluate(baseline, X_test, y_test)
        metrics["cv_roc_auc"] = cv_roc_auc(baseline, X_train, y_train)

        mlflow.log_params({"model_type": "logistic_regression", "approach": "baseline",
                           "C": 1.0, "scaler": "StandardScaler"})
        mlflow.log_metrics(metrics)
        log_confusion_matrix(baseline, X_test, y_test, "baseline")
        mlflow.sklearn.log_model(baseline, name="model", input_example=X_train.head(2),
                                 serialization_format=CLOUDPICKLE)
        results["baseline"] = (run.info.run_id, metrics)
        logger.info("baseline " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # ---------- 2. AutoML (FLAML) ----------
    with mlflow.start_run(run_name="automl_flaml") as run:
        automl = AutoML()
        automl.fit(X_train, y_train, task="classification", metric="roc_auc",
                   time_budget=budget, estimator_list=["lgbm", "rf", "xgboost",
                   "extra_tree", "lrl1"], seed=42, verbose=0,
                   mlflow_logging=False)   # we log the final result ourselves
        best = automl.model.estimator      # the underlying fitted sklearn-style model
        metrics = evaluate(best, X_test, y_test)
        metrics["cv_roc_auc"] = cv_roc_auc(best, X_train, y_train)

        mlflow.log_params({"approach": "automl", "automl_tool": "FLAML",
                           "time_budget_s": budget, "search_metric": "roc_auc",
                           "best_estimator": automl.best_estimator})
        # the hyperparameters FLAML chose
        mlflow.log_params({f"hp_{k}": v for k, v in automl.best_config.items()})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("flaml_val_roc_auc", 1 - automl.best_loss)
        mlflow.log_metric("flaml_trials", len(automl.config_history))
        log_confusion_matrix(best, X_test, y_test, "automl")
        mlflow.sklearn.log_model(best, name="model", input_example=X_train.head(2),
                                 serialization_format=CLOUDPICKLE)
        results["automl"] = (run.info.run_id, metrics)
        logger.info(f"automl best_estimator={automl.best_estimator} "
                    + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # ---------- 3. Compare and register the winner ----------
    print("\n{:<10}".format("model") + "".join(f"{m:>11}" for m in results["baseline"][1]))
    for name, (_, m) in results.items():
        print(f"{name:<10}" + "".join(f"{v:>11.4f}" for v in m.values()))

    # Winner = higher cross-validated ROC-AUC, then test ROC-AUC as tie-breaker.
    # Both models are registered: winner gets alias 'champion' (served),
    # the other gets 'challenger' (kept for comparison / rollback).
    ranked = sorted(results, key=lambda k: (results[k][1]["cv_roc_auc"],
                                            results[k][1]["roc_auc"]), reverse=True)
    client = MlflowClient()
    for name, alias in zip(ranked, (MODEL_ALIAS, "challenger")):
        mv = mlflow.register_model(f"runs:/{results[name][0]}/model", MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
        client.set_model_version_tag(MODEL_NAME, mv.version, "approach", name)
        logger.info(f"registered {name} as {MODEL_NAME} v{mv.version} alias=@{alias}")
        print(f"{name:<10} -> {MODEL_NAME} version {mv.version} (@{alias})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=60, help="AutoML time budget (s)")
    main(parser.parse_args().budget)
