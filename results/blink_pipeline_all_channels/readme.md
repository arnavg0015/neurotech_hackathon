# 21-channel blink model

This is the 21-channel blink random forest used for channel analysis. `blink_classifier.pkl` holds the model, and the other files hold epochs, processed recordings, timing, splits, and results. It does not replace the demo's two-channel model.

To rebuild these files, run [blink_pipeline.ipynb](../../training/blink_pipeline.ipynb) with the `neurotech` kernel. To inspect channel importance, run [feature_importance.ipynb](../../feature_importance/feature_importance.ipynb).
