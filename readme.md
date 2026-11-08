# Evoke

This project uses EEG recordings to detect the spoken word “activate” and double blinks. In the demo, activation starts a rotating arrow and a double blink switches the selected appliance on or off.

## Demo

https://github.com/user-attachments/assets/70374ab7-2611-48ac-8f31-bc60ae293d9a

## Setup

Run these commands from the project folder:

```bash
conda env create -f environment.yml
conda activate neurotech
python demo/demo.py
```

Open http://127.0.0.1:8000. The demo uses saved models from `results/` and a curated one-minute activation replay in `demo/replay/`. Blink selection adds to the playback time.

## Folders

- [demo](demo/readme.md): the EEG replay and appliance selector.
- [stimulation](stimulation/readme.md): the screen shown during data collection.
- [training](training/readme.md): activation and blink training notebooks.
- [feature_importance](feature_importance/readme.md): channel importance and the activation channel-count comparison.
- [data](data/readme.md): raw EEG recordings and stimulus logs.
- [results](results/readme.md): saved models, processed recordings, predictions, and plots.

