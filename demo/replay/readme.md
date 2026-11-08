# Curated replay

`activation.npz` contains 60 seconds of processed EEG at 500 Hz, plus 39 original one-second model inputs. All 21 channels are in microvolts. `activation.json` records the six source clips, their event times, preprocessing settings, and file hashes. `predictions.csv` holds the matching saved predictions.

The clips come from spoken Trial 5, around activation events 8, 50, 90, 102, 136, and 178. They were selected using known model outcomes for the demo. The original recordings and test results are unchanged. Dashed lines in the chart mark joins between clips.

From the project root, run `python demo/prepare_replay.py` in the `neurotech` environment to rebuild these files. The blink stage still uses the original recording in [blink_pipeline](../../results/blink_pipeline/readme.md).
