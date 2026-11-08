"""Build the one-minute demo from saved Trial 5 signals and epochs."""

import hashlib
import json

import mne
import numpy as np
import pandas as pd

from demo import DEMO_DIR, ROOT, Replay


def prepare():
    folder = ROOT / "results/hackathon_pipeline"
    output = DEMO_DIR / "replay"
    targets = [8, 50, 90, 102, 136, 178]
    replay = Replay("activation", curated=False)
    report = json.loads((folder / "metrics.json").read_text())
    source = next(row for row in report["sources"] if row["trial"] == 5)
    log = pd.read_csv(ROOT / source["stimulus_file"], keep_default_na=False)
    raw_path = folder / "trial_5_preprocessed_raw.fif"
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="error")
    sfreq = int(raw.info["sfreq"])
    origin_ms = float(log["onset_epoch_ms"].iloc[0])
    origin_in_eeg = origin_ms / 1000 - pd.Timestamp(raw.info["meas_date"]).timestamp() - 15.6
    rows = pd.read_csv(folder / "epoch_split.csv")
    rows = rows[rows["trial"].eq(5)].set_index("event_id")
    saved = pd.read_csv(folder / "test_predictions.csv").set_index("event_id")
    signals, epochs, ids, events, clips = [], [], [], [], []

    for index, target in enumerate(targets):
        first = int(rows.loc[target, "start_sample"]) - 8 * sfreq
        stop = first + 10 * sfreq
        assert 0 <= first < stop <= raw.n_times
        assert not clips or first >= clips[-1]["source_stop_sample"]
        start, end = index * 10, (index + 1) * 10
        source_start = first / sfreq - origin_in_eeg
        source_end = stop / sfreq - origin_in_eeg
        kept = rows[(rows["start_sample"] >= first) & (rows["start_sample"] + sfreq <= stop)]
        assert kept["label"].sum() == 1 and kept.loc[target, "label"] == 1
        signals.append(raw.get_data(start=first, stop=stop) * 1e6)
        for event_id in kept.index:
            ids.append(int(event_id))
            epochs.append(replay.inputs[event_id][0])
        for original in replay.events.values():
            if original["end"] <= source_start or original["start"] >= source_end:
                continue
            event = dict(original, clip=index + 1,
                         start=start + max(0, original["start"] - source_start),
                         end=min(end, start + original["end"] - source_start),
                         input_start=None, ready_at=None)
            if original["id"] in kept.index:
                event["input_start"] = start + (int(kept.loc[original["id"], "start_sample"]) - first) / sfreq
                event["ready_at"] = event["input_start"] + 1
                assert event["start"] - 1e-6 <= event["input_start"] < event["ready_at"] <= event["end"] + 1e-6
            events.append(event)
        clips.append({"start": start, "end": end, "activate_event": target,
                      "source_start_sample": first, "source_stop_sample": stop,
                      "source_start_s": source_start, "source_end_s": source_end})

    signal, X = np.concatenate(signals, axis=1), np.stack(epochs)
    y = saved.loc[ids, "label"].to_numpy()
    predicted = replay.model.predict(X)
    assert signal.shape == (21, 30000) and X.shape == (39, 21, 500)
    assert np.isfinite(signal).all() and np.isfinite(X).all()
    assert len(set(event["id"] for event in events)) == len(events)
    assert [event["id"] for event in events if event["word"] == "activate"] == targets
    assert np.array_equal(predicted, saved.loc[ids, "predicted"])
    assert np.array_equal(np.array(ids)[predicted == 1], targets)
    model_path = folder / "activate_classifier.pkl"
    sources = [model_path, raw_path, folder / "epochs.npz", folder / "epoch_split.csv", ROOT / source["stimulus_file"]]
    metadata = {
        "description": "Curated demonstration selected using known model outcomes; not a new test evaluation.",
        "duration": 60, "trial": 5, "sample_rate": sfreq, "channels": raw.ch_names,
        "units": "microvolts", "preprocessing": report["preprocessing"],
        "source_origin_in_eeg_s": origin_in_eeg, "source_stimulus_origin_epoch_ms": origin_ms,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "epoch_policy": "Reuse original baseline-corrected epochs; omit windows cut by clip boundaries.",
        "clips": clips, "events": events,
    }
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "activation.npz", signal_uv=signal, X=X, event_id=ids,
                        y=y, channels=raw.ch_names, sfreq=sfreq)
    (output / "activation.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    saved.loc[ids].assign(predicted=predicted).to_csv(output / "predictions.csv")
    print(f"Saved 60 seconds, {len(ids)} epochs, and {int(predicted.sum())} activations to {output}")


if __name__ == "__main__":
    prepare()
