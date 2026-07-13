# ⚽ FIFA Player Position Predictor

An interactive demo that predicts a football player's best-fit outfield position —
**Forward, Midfielder, or Defender** — from seven core attributes, with a
confidence score for each class. Move the sliders (or pick a preset like *Striker*
or *Playmaker*) and the model classifies the player instantly.

**Live demo:** _(Streamlit Community Cloud link goes here once deployed)_

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
