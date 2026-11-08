# Activation model

This is the 21-channel activation random forest used by the [demo](../../demo/readme.md). `activate_classifier.pkl` holds the model, `epochs.npz` holds its inputs, and `epoch_split.csv` identifies their trials and events. The `.fif` files hold processed recordings, epochs, and ICA fits; the CSV and JSON files hold checks and results.

To rebuild these files, run [eeg_pipeline.ipynb](../../training/eeg_pipeline.ipynb) with the `neurotech` kernel. The demo uses the saved Trial 5 outputs.
