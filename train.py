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
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "players_22.csv"
# Public mirror of the Kaggle "FIFA 22 complete player dataset".
DATA_URL = (
    "https://raw.githubusercontent.com/abineshta/"
    "FIFA-22-complete-player-dataset-EDA/main/players_22.csv"
)
OUT = HERE / "model"
OUT.mkdir(exist_ok=True)


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
    print(f"\n[train] test accuracy = {acc:.4f}\n")
    print(classification_report(y_te, pred, target_names=CLASSES, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, pred))

    importances = sorted(
        zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1]
    )
    print("\n[train] feature importance:")
    for name, imp in importances:
        print(f"   {name:24s} {imp:.3f}")

    joblib.dump(
        {"model": clf, "features": FEATURES, "classes": CLASSES},
        OUT / "model.pkl", compress=3,
    )
    metrics = {
        "accuracy": round(float(acc), 4),
        "n_samples": int(len(X)),
        "features": FEATURES,
        "classes": CLASSES,
        "importance": {n: round(float(i), 4) for n, i in importances},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[train] saved -> {OUT/'model.pkl'}  and  {OUT/'metrics.json'}")


if __name__ == "__main__":
    main()
