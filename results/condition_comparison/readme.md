# Condition comparison

`test_accuracy.png` and `test_accuracy.svg` are the 16:9 plots for slides. They show test accuracy for random forest, SVM, logistic regression, and LDA under four training conditions.

| Condition | Train | Test |
|---|---|---|
| Thought-only | Experiment 1, Trials 1–2 | Experiment 1, Trial 3 |
| Spoken, less enunciation | Experiment 2, Trial 1 | Experiment 2, Trial 5 |
| Spoken, more enunciation | Experiment 2, Trials 2–3 | Experiment 2, Trial 5 |
| Spoken, all | Experiment 2, Trials 1–4 | Experiment 2, Trial 5 |

`test_accuracy.csv` contains the plot values as fractions from 0 to 1. `metrics.csv` adds epoch counts and the always-not-activate comparison. Predictions, class counts, and the exact splits are in their named CSV files. The 16 `.pkl` files hold the fitted pipelines and accept one-second, 21-channel EEG epochs in microvolts at 500 Hz.

`mental_epochs.npz` and `mental_epoch_metadata.csv` contain the new mental epochs. The spoken epochs come from [hackathon_pipeline](../hackathon_pipeline/readme.md). `metadata.json`, `preprocessing_qc.csv`, and `channel_checks.csv` record the preprocessing, source hashes, resumed Trial 2 timing, and model settings.

Run [condition_comparison.ipynb](../../training/condition_comparison.ipynb) with the `neurotech` kernel to rebuild these files. The existing demo models are unchanged. Trial 5 was previously inspected, and training sizes differ, so these are exploratory results rather than a controlled comparison of enunciation alone.
