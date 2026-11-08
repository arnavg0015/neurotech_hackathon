# Training

- [eeg_pipeline.ipynb](eeg_pipeline.ipynb) trains activation detection on the spoken recordings. It trains on Trials 1–3, validates on Trial 4, refits on Trials 1–4, and tests on Trial 5. It saves the model used by the demo to [hackathon_pipeline](../results/hackathon_pipeline/readme.md).
- [blink_pipeline.ipynb](blink_pipeline.ipynb) trains a double-blink model. It trains on Trial 1, validates on Trial 2, refits on Trials 1–2, and tests on Trial 3. It saves to [blink_pipeline_all_channels](../results/blink_pipeline_all_channels/readme.md). 
- [condition_comparison.ipynb](condition_comparison.ipynb) compares four classifiers trained on thought-only, less-enunciated speech, more-enunciated speech, and all spoken trials. It saves the test-accuracy plot and separate models to [condition_comparison](../results/condition_comparison/readme.md). Select the `neurotech` kernel and run the cells in order.

