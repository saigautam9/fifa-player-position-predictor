"""
FIFA Player Position Predictor — Streamlit demo.

Set a player's attributes and the model predicts their best-fit outfield
position (Forward / Midfielder / Defender) with a confidence for each class.

Model: scikit-learn RandomForest retrained on the public FIFA 22 player dataset
(the original Spark ML artifact was not preserved). Same inputs and the same
three classes as the original project, including its two engineered features.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

HERE = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    bundle = joblib.load(HERE / "model" / "model.pkl")
    metrics = json.loads((HERE / "model" / "metrics.json").read_text())
    return bundle["model"], bundle["classes"], metrics


MODEL, CLASSES, METRICS = load_model()
ACC = METRICS["accuracy"]

EMOJI = {"Forward": "⚽", "Midfielder": "🎯", "Defender": "🛡️"}
COLOR = {"Forward": "#2ca25f", "Midfielder": "#4575b4", "Defender": "#d97706"}

KEYS = ["pace", "stamina", "shooting", "passing", "finishing", "defending", "tackling"]
LABELS = {
    "pace": "Pace 🏃", "stamina": "Stamina 🔋", "shooting": "Shooting 🎯",
    "passing": "Passing 🅿️", "finishing": "Finishing ⚽",
    "defending": "Defending 🛡️", "tackling": "Tackling 🦵",
}
DEFAULTS = {"pace": 75, "stamina": 70, "shooting": 60, "passing": 65,
            "finishing": 55, "defending": 55, "tackling": 55}

# name -> [pace, stamina, shooting, passing, finishing, defending, tackling]
PRESETS = {
    "⚽ Striker": [89, 77, 91, 65, 94, 45, 40],
    "🎯 Playmaker": [74, 78, 86, 93, 82, 64, 61],
    "🛡️ Centre-back": [79, 84, 60, 71, 45, 90, 89],
    "🏃 Full-back": [76, 84, 66, 89, 55, 80, 79],
}


def engineer(v: dict) -> np.ndarray:
    defense_strength = 0.5 * v["defending"] + 0.3 * v["tackling"] + 0.2 * v["stamina"]
    attack_to_defense_ratio = (
        0.5 * v["shooting"] + 0.3 * v["passing"] + 0.2 * v["finishing"]
    ) / (v["defending"] + 1)
    return np.array([[
        v["pace"], v["stamina"], v["shooting"], v["passing"], v["finishing"],
        v["defending"], v["tackling"], defense_strength, attack_to_defense_ratio,
    ]])


st.set_page_config(page_title="FIFA Player Position Predictor", page_icon="⚽", layout="wide")

for k, d in DEFAULTS.items():
    st.session_state.setdefault(k, d)

st.title("⚽ FIFA Player Position Predictor")
st.markdown(
    "Set a player's attributes and the model predicts their best-fit outfield "
    "position — **Forward, Midfielder, or Defender** — with confidence scores."
)
st.caption(
    f"Random Forest · trained on the public FIFA 22 dataset (~17k players) · "
    f"**{ACC:.1%}** test accuracy · 3 classes · retrained scikit-learn model."
)

# --- presets: clicking one sets the sliders (handled before sliders render) ---
st.write("**Try a preset**")
pcols = st.columns(len(PRESETS))
for (name, vals), col in zip(PRESETS.items(), pcols):
    if col.button(name, use_container_width=True):
        for k, val in zip(KEYS, vals):
            st.session_state[k] = val

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Attributes")
    for k in KEYS:
        st.slider(LABELS[k], 0, 99, key=k)

vals = {k: st.session_state[k] for k in KEYS}
proba = MODEL.predict_proba(engineer(vals))[0]
order = np.argsort(proba)[::-1]
top = CLASSES[order[0]]

with right:
    st.subheader("Prediction")
    st.markdown(
        f"<div style='font-size:2.2rem;font-weight:700;color:{COLOR[top]}'>"
        f"{EMOJI[top]} {top}</div>"
        f"<div style='color:gray'>{proba[order[0]]:.0%} confident</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    for i in order:
        cls = CLASSES[i]
        st.markdown(f"{EMOJI[cls]} **{cls}** — {proba[i]:.0%}")
        st.progress(float(proba[i]))

st.divider()
st.caption(
    "Two engineered features from the original project — `defense_strength` and "
    "`attack_to_defense_ratio` — are computed from the sliders before prediction; "
    "`attack_to_defense_ratio` is the model's most important input. "
    "The original project used a Spark ML model that wasn't preserved, so this "
    "demo is an equivalent model retrained on the same public data."
)
