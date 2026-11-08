# EEG Detection

This project uses EEG recordings to detect the spoken word “activate” and double blinks. The demo replays two recordings: activation starts a rotating arrow, and a double blink selects a bulb.

## Setup

Run these commands from the project folder:

```bash
conda env create -f environment.yml
conda activate neurotech
python demo/demo.py
```

Open http://127.0.0.1:8000. The demo uses the saved models and recordings already in `results/`.

## Folders

- [demo](demo/readme.md): the EEG replay and bulb selector.
- [stimulation](stimulation/readme.md): the screen shown during data collection.
- [training](training/readme.md): activation and blink training notebooks.
- [feature_importance](feature_importance/readme.md): channel importance and the activation channel-count comparison.
- [data](data/readme.md): raw EEG recordings and stimulus logs.
- [results](results/readme.md): saved models, processed recordings, predictions, and plots.

