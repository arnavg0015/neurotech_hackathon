# Channel analysis

This folder contains channel rankings, permutation results, and plots for both models. `activation_channel_sweep.csv` compares the top 1–21 channels; `activation_channel_sweep_predictions.csv` contains their Trial 5 predictions.

Run [feature_importance.ipynb](../../feature_importance/feature_importance.ipynb) with the `neurotech` kernel to regenerate the channel results. Trial 5 supplies both the activation ranking and evaluation, so the comparison is exploratory.

Files named `*_features.png`, `*_feature_ranking.csv`, and `*_feature_repeats.csv` are older individual-feature analyses. The current notebook focuses on whole channels and does not regenerate them.
