# ml-baseball-strike-zone

Predicting MLB player strike zones with a Support Vector Machine, from raw pitch-by-pitch data.

## What it does

Trains an SVM classifier to predict whether a given pitch is a strike or a ball for individual MLB players, using pitch location data from real games. The output is a per-player decision boundary you can visualize as the player's "personal strike zone" — useful for thinking about how umpires and pitchers actually see different hitters.

## How to run

The working code lives in [`predict-baseball-strike-zones-with-machine-learning/`](./predict-baseball-strike-zones-with-machine-learning/):

```bash
cd predict-baseball-strike-zones-with-machine-learning

# install + download player pitch data
python3 setup_and_download.py

# open the notebook
jupyter notebook predict_baseball_strike_zones_with_svm.ipynb
```

Full setup instructions (including a NumPy compatibility note) are in [`SETUP_GUIDE.md`](./predict-baseball-strike-zones-with-machine-learning/SETUP_GUIDE.md).

## Tech

Python · scikit-learn · pandas · matplotlib · Jupyter

## Data

Pitch-by-pitch data is fetched at setup time from public MLB sources (see `download_player_data.py`). No data is committed to the repo.
