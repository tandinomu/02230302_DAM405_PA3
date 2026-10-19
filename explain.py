"""
Explain the registered model with SHAP.

* Loads the 'champion' model from the MLflow Model Registry.
* Computes SHAP values on the test set (model-agnostic explainer, so it works
  whichever model AutoML picked).
* Saves a bar plot (global importance) and a beeswarm plot (direction of effect),
  prints the top features and logs everything to a new MLflow run.

Run:  python explain.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap

from config import ARTIFACT_DIR, MODEL_ALIAS, MODEL_NAME, get_logger, setup_mlflow
from train import load_data

logger = get_logger("explain")


def main():
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_data()
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    logger.info(f"explaining {MODEL_NAME}@{MODEL_ALIAS} ({type(model).__name__})")

    # Explain the probability of class 1 (benign). A small background sample
    # keeps the computation fast.
    predict_benign = lambda data: model.predict_proba(
        pd.DataFrame(data, columns=X_train.columns))[:, 1]
    background = shap.sample(X_train, 100, random_state=42)
    explainer = shap.Explainer(predict_benign, background, seed=42)
    sv = explainer(X_test)

    # ---- global importance: mean |SHAP| per feature ----
    importance = (pd.DataFrame({"feature": X_test.columns,
                                "mean_abs_shap": np.abs(sv.values).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False))
    imp_path = os.path.join(ARTIFACT_DIR, "shap_importance.csv")
    importance.to_csv(imp_path, index=False)
    print("\nTop 10 features by mean |SHAP| (impact on P(benign)):")
    print(importance.head(10).to_string(index=False))

    # ---- plots ----
    shap.plots.bar(sv, max_display=10, show=False)
    plt.title("Global feature importance (mean |SHAP|)")
    bar_path = os.path.join(ARTIFACT_DIR, "shap_bar.png")
    plt.savefig(bar_path, bbox_inches="tight", dpi=130)
    plt.close()

    shap.plots.beeswarm(sv, max_display=10, show=False)
    plt.title("SHAP values: how each feature moves P(benign)")
    bee_path = os.path.join(ARTIFACT_DIR, "shap_beeswarm.png")
    plt.savefig(bee_path, bbox_inches="tight", dpi=130)
    plt.close()

    # ---- log to MLflow ----
    with mlflow.start_run(run_name="shap_explanation"):
        mlflow.log_param("explained_model", f"{MODEL_NAME}@{MODEL_ALIAS}")
        mlflow.log_param("background_size", 100)
        for _, row in importance.head(5).iterrows():
            mlflow.log_metric(f"shap_{row.feature.replace(' ', '_')}", row.mean_abs_shap)
        for p in (imp_path, bar_path, bee_path):
            mlflow.log_artifact(p, artifact_path="explainability")
    logger.info(f"SHAP plots saved to {ARTIFACT_DIR} and logged to MLflow")


if __name__ == "__main__":
    main()
