"""Draws the workflow diagram (docs/architecture.png) used in the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 7.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.4); ax.axis("off")
EDGE = "#3A3A3A"


def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=EDGE, lw=1.2))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top",
            fontsize=11, fontweight="bold", color="#1F1F1F")
    ax.text(x + w / 2, y + h - 0.65, "\n".join(lines), ha="center", va="top",
            fontsize=8.6, color="#333333", linespacing=1.35)


def arrow(p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.4, color=EDGE))


W, H = 2.6, 2.2
xs = [0.3, 3.45, 6.6, 9.75]            # four columns, gap 0.55
top, bot = 4.5, 1.1

ax.text(0.3, 7.1, "Offline: train, track, explain", fontsize=10.5,
        color="#555555", fontweight="bold")
box(xs[0], top, W, H, "1. Data", ["Breast cancer dataset", "569 samples, 30 features",
    "80/20 stratified split", "training set saved as", "drift reference"], "#E8EEF7")
box(xs[1], top, W, H, "2. Train (train.py)", ["Baseline: scaled", "logistic regression",
    "AutoML: FLAML, 60 s", "lgbm, rf, xgboost,", "extra_tree, lrl1"], "#DCEBDD")
box(xs[2], top, W, H, "3. MLflow", ["Tracking: params, metrics,", "plots, models",
    "Registry: winner", "@champion, other", "@challenger"], "#FDEBD3")
box(xs[3], top, W, H, "4. Explain (explain.py)", ["SHAP on @champion", "bar + beeswarm plots",
    "top features logged", "back to MLflow"], "#EADCF0")
for a, b in zip(xs[:-1], xs[1:]):
    arrow((a + W, top + H / 2), (b, top + H / 2))

ax.text(0.3, 3.65, "Online: serve, log, monitor", fontsize=10.5,
        color="#555555", fontweight="bold")
box(xs[0], bot, W, H, "Clients", ["test_client.py", "normal traffic", "bad requests",
    "drifted traffic"], "#E8EEF7")
box(xs[1], bot, W, H, "5. Service (app.py)", ["Flask POST /predict", "loads @champion",
    "GET /stats, GET /drift", "returns label +", "probability"], "#D9ECF2")
box(xs[2], bot, W, H, "6. Monitoring", ["request count, error rate", "avg + p95 latency",
    "KS test per feature", "vs training reference"], "#F7DADA")
box(xs[3], bot, W, H, "7. Logs", ["logs/service.log", "timestamp | level |", "logger | message",
    "rotating, 1 MB x 3"], "#EFEFEF")
for a, b in zip(xs[:-1], xs[1:]):
    arrow((a + W, bot + H / 2), (b, bot + H / 2))

# MLflow registry -> service (elbow: down, left, down)
ax.plot([xs[2] + W / 2, xs[2] + W / 2, xs[1] + W / 2], [top, 3.75, 3.75], color=EDGE, lw=1.4)
arrow((xs[1] + W / 2, 3.75), (xs[1] + W / 2, bot + H))
ax.text((xs[1] + xs[2]) / 2 + W / 2, 3.85, "load model @champion", ha="center",
        fontsize=8.5, color="#444444", style="italic")

ax.text(6.5, 0.55, "Levels: DEBUG payload | INFO prediction, latency, summary | "
        "WARNING out-of-range input, drift | ERROR bad request | CRITICAL model missing",
        ha="center", fontsize=8.2, color="#555555", style="italic")
plt.savefig("docs/architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
print("saved docs/architecture.png")
