# ⚽ FIFA Player Position Predictor

An interactive demo that predicts a football player's best-fit outfield position —
**Forward, Midfielder, or Defender** — from seven core attributes, with a
confidence score for each class. Move the sliders (or pick a preset like *Striker*
or *Playmaker*) and the model classifies the player instantly.

### 🔗 Live demo → **[fifa-player-position-predictor.streamlit.app](https://fifa-player-position-predictor-8yklyqsvsavrpwepjnmzfd.streamlit.app/)**

## The model

A **scikit-learn Random Forest** trained on the public
[FIFA 22 player dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)
(~17,000 outfield players). It reaches **82.6% test accuracy** across the three
classes — Forwards and Defenders are almost never confused; the residual error
sits, as expected, on the Midfielder boundary.

**Inputs (0–99):** pace, stamina, shooting, passing, finishing, defending, tackling.

**Engineered features** (computed from the inputs, carried over from the original
project design):

- `defense_strength = 0.5·defending + 0.3·tackling + 0.2·stamina`
- `attack_to_defense_ratio = (0.5·shooting + 0.3·passing + 0.2·finishing) / (defending + 1)`

`attack_to_defense_ratio` turns out to be the model's most important single
feature.

## Evaluation

All numbers below are produced by `python train.py` and written to
[`model/metrics.json`](model/metrics.json). Trained on **13,685** players,
evaluated on a held-out **3,422**-player test set (80/20 stratified split, seed 42).

| Metric | Score |
| --- | --- |
| Accuracy | **82.6%** |
| Macro F1 | **0.824** |
| Weighted F1 | **0.826** |

### Per-class

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| ⚽ Forward | 0.759 | 0.852 | **0.803** | 736 |
| 🎯 Midfielder | 0.818 | 0.746 | **0.780** | 1,407 |
| 🛡️ Defender | 0.877 | 0.901 | **0.889** | 1,279 |

### Confusion matrix (rows = actual, cols = predicted)

| | → Forward | → Midfielder | → Defender |
| --- | --- | --- | --- |
| **Forward** | 627 | 107 | 2 |
| **Midfielder** | 199 | 1049 | 159 |
| **Defender** | 0 | 127 | 1152 |

The errors are football-sensible: Forwards and Defenders are almost never
confused (only 2 of 736 Forwards mislabelled Defender), and the residual
confusion sits on the Midfielder boundary — the role that genuinely overlaps
with both attack and defence.

### Feature importance

| Feature | Importance |
| --- | --- |
| `attack_to_defense_ratio` | **0.310** |
| tackling | 0.132 |
| `defense_strength` | 0.112 |
| finishing | 0.106 |
| defending | 0.085 |
| shooting | 0.080 |
| passing | 0.078 |
| pace | 0.068 |
| stamina | 0.028 |

The engineered `attack_to_defense_ratio` is the single strongest predictor —
validating the original project's feature design.

## Note on the model

The original project trained a **Spark ML** Random Forest; that artifact was not
preserved, so this demo uses an **equivalent scikit-learn model retrained on the
same public data with the same inputs and classes**. The reported accuracy is this
retrained model's own measured test accuracy.

## Run locally

```bash
pip install -r requirements.txt
python train.py            # regenerates model/model.pkl (auto-downloads the data)
streamlit run streamlit_app.py
```

## Deploy (free)

Hosted on **Streamlit Community Cloud**: push this repo to GitHub, then at
[share.streamlit.io](https://share.streamlit.io) sign in with GitHub, pick the
repo, and set the main file to `streamlit_app.py`.
