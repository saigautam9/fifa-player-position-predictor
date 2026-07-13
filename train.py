"""
Train the FIFA player-position classifier.

The original project trained a Spark ML RandomForest on a teammate's machine and
that artifact is gone, so this retrains an equivalent scikit-learn RandomForest on
the public FIFA 22 player dataset. Same task, same inputs, same three classes.

Inputs the demo collects (identical to app.py):
    pace, stamina, shooting, passing, finishing, defending, tackling
plus the two engineered features from the original app:
    defense_strength        = 0.5*defending + 0.3*tackling + 0.2*stamina
    attack_to_defense_ratio = (0.5*shooting + 0.3*passing + 0.2*finishing) / (defending + 1)

Target (primary position -> 3 classes):  Forward / Midfielder / Defender
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import label_binarize  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "players_22.csv"
# Public mirror of the Kaggle "FIFA 22 complete player dataset".
DATA_URL = (
    "https://raw.githubusercontent.com/abineshta/"
    "FIFA-22-complete-player-dataset-EDA/main/players_22.csv"
)
OUT = HERE / "model"
OUT.mkdir(exist_ok=True)
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

# Colorblind-safe (Okabe-Ito), matching the app's per-class colors.
CLASS_COLORS = {"Forward": "#009E73", "Midfielder": "#0072B2", "Defender": "#D55E00"}


def plot_confusion(cm, path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=15)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — test set")
    thresh = cm.max() / 2
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_roc(y_te, proba, path) -> None:
    yb = label_binarize(y_te, classes=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
        ax.plot(fpr, tpr, color=CLASS_COLORS[c], lw=2,
                label=f"{c} (AUC {roc_auc_score(yb[:, i], proba[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — one-vs-rest")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_importance(importances, path) -> None:
    names = [n for n, _ in importances][::-1]
    vals = [v for _, v in importances][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.barh(names, vals, color="#0072B2")
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title("Feature importance")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def ensure_data() -> None:
    if DATA.exists():
        return
    DATA.parent.mkdir(exist_ok=True)
    print(f"[train] downloading FIFA 22 dataset -> {DATA} ...")
    urllib.request.urlretrieve(DATA_URL, DATA)
    print("[train] download complete")

# Base inputs, in the exact order the demo will pass them.
BASE_FEATURES = ["pace", "stamina", "shooting", "passing", "finishing", "defending", "tackling"]
FEATURES = BASE_FEATURES + ["defense_strength", "attack_to_defense_ratio"]
CLASSES = ["Forward", "Midfielder", "Defender"]  # index == label id, matches app.py position_map

# First listed position -> coarse class. Goalkeepers are dropped (app has no GK class).
POSITION_TO_CLASS = {
    "ST": "Forward", "CF": "Forward", "LW": "Forward", "RW": "Forward",
    "LF": "Forward", "RF": "Forward",
    "CM": "Midfielder", "CDM": "Midfielder", "CAM": "Midfielder",
    "LM": "Midfielder", "RM": "Midfielder",
    "CB": "Defender", "LB": "Defender", "RB": "Defender",
    "LWB": "Defender", "RWB": "Defender",
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["defense_strength"] = 0.5 * df["defending"] + 0.3 * df["tackling"] + 0.2 * df["stamina"]
    df["attack_to_defense_ratio"] = (
        0.5 * df["shooting"] + 0.3 * df["passing"] + 0.2 * df["finishing"]
    ) / (df["defending"] + 1)
    return df


def load() -> tuple[pd.DataFrame, pd.Series]:
    ensure_data()
    raw = pd.read_csv(DATA, low_memory=False)

    # Map raw columns to the app's input names.
    df = pd.DataFrame({
        "pace": raw["pace"],
        "stamina": raw["power_stamina"],
        "shooting": raw["shooting"],
        "passing": raw["passing"],
        "finishing": raw["attacking_finishing"],
        "defending": raw["defending"],
        "tackling": (raw["defending_standing_tackle"] + raw["defending_sliding_tackle"]) / 2.0,
    })

    primary = raw["player_positions"].str.split(",").str[0].str.strip()
    df["label"] = primary.map(POSITION_TO_CLASS)

    # Drop goalkeepers / unmapped and any row with a missing attribute.
    df = df.dropna(subset=BASE_FEATURES + ["label"])
    df = engineer(df)

    y = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    return df[FEATURES], y


def main() -> None:
    X, y = load()
    print(f"[train] samples={len(X):,}  features={len(FEATURES)}")
    dist = y.value_counts().sort_index()
    print("[train] class balance:", {CLASSES[i]: int(dist.get(i, 0)) for i in range(len(CLASSES))})

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Depth-capped, moderate forest: keeps accuracy but stays a small artifact
    # (an unbounded forest here balloons to ~70 MB, needlessly).
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    # Fit on plain arrays so inference with a numpy vector needs no pandas and
    # raises no feature-name warning.
    clf.fit(X_tr.values, y_tr)

    pred = clf.predict(X_te.values)
    acc = accuracy_score(y_te, pred)
    report = classification_report(y_te, pred, target_names=CLASSES, digits=3, output_dict=True)
    cm = confusion_matrix(y_te, pred)
    print(f"\n[train] test accuracy = {acc:.4f}\n")
    print(classification_report(y_te, pred, target_names=CLASSES, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)

    importances = sorted(
        zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1]
    )
    print("\n[train] feature importance:")
    for name, imp in importances:
        print(f"   {name:24s} {imp:.3f}")

    # --- additional metrics ---------------------------------------------------
    labels_idx = list(range(len(CLASSES)))
    proba = clf.predict_proba(X_te.values)
    roc_auc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=labels_idx)
    ll = log_loss(y_te, proba, labels=labels_idx)
    kappa = cohen_kappa_score(y_te, pred)
    mcc = matthews_corrcoef(y_te, pred)

    print("\n[train] 5-fold stratified cross-validation ...")
    cv = cross_validate(
        RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_leaf=5,
                               class_weight="balanced", n_jobs=-1, random_state=42),
        X.values, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring=["accuracy", "f1_macro"], n_jobs=-1,
    )
    cv_acc, cv_f1 = cv["test_accuracy"], cv["test_f1_macro"]
    print(f"   ROC-AUC (OvR macro) : {roc_auc:.4f}")
    print(f"   Log loss           : {ll:.4f}")
    print(f"   Cohen's kappa      : {kappa:.4f}")
    print(f"   Matthews corrcoef  : {mcc:.4f}")
    print(f"   CV accuracy        : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"   CV macro-F1        : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    plot_confusion(cm, ASSETS / "confusion_matrix.png")
    plot_roc(y_te, proba, ASSETS / "roc_curves.png")
    plot_importance(importances, ASSETS / "feature_importance.png")
    print(f"[train] saved plots -> {ASSETS}/")

    joblib.dump(
        {"model": clf, "features": FEATURES, "classes": CLASSES},
        OUT / "model.pkl", compress=3,
    )
    metrics = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "roc_auc_ovr_macro": round(float(roc_auc), 4),
        "log_loss": round(float(ll), 4),
        "cohen_kappa": round(float(kappa), 4),
        "matthews_corrcoef": round(float(mcc), 4),
        "cv_5fold": {
            "accuracy_mean": round(float(cv_acc.mean()), 4),
            "accuracy_std": round(float(cv_acc.std()), 4),
            "macro_f1_mean": round(float(cv_f1.mean()), 4),
            "macro_f1_std": round(float(cv_f1.std()), 4),
        },
        "n_samples": int(len(X)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "features": FEATURES,
        "classes": CLASSES,
        "per_class": {
            c: {
                "precision": round(float(report[c]["precision"]), 3),
                "recall": round(float(report[c]["recall"]), 3),
                "f1": round(float(report[c]["f1-score"]), 3),
                "support": int(report[c]["support"]),
            }
            for c in CLASSES
        },
        "confusion_matrix": cm.tolist(),
        "importance": {n: round(float(i), 4) for n, i in importances},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[train] saved -> {OUT/'model.pkl'}  and  {OUT/'metrics.json'}")


if __name__ == "__main__":
    main()
