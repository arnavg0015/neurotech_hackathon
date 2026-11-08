"""
Align stimulus CSVs to the preprocessed EEG, cut epochs, and build the dataset.

The EEG and stimulus ran on two laptops with no shared triggers. Timing comes
from each laptop's clock plus STIM_CLOCK_AHEAD_S, estimated from speech
muscle artifacts (speech_lag.py).

Per recording -> results/preproc/<name>/
    <name>-epo.fif         epochs, CSV rows attached as metadata
    <name>_labels.csv      one row per epoch
    <name>_alignment.json  timing info

Combined dataset -> results/dataset/
    exp1_mental-epo.fif / exp2_spoken-epo.fif   all trials of each experiment
    exp1_mental.npz / exp2_spoken.npz           X (epochs, channels, times), y, groups
    exp1_mental_labels.csv / ...                metadata for every epoch
"""
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_processed"
SAVE = ROOT / "results" / "preproc"
DATASET = ROOT / "results" / "dataset"


def main():
    # ------------------------------------------------------------------ flags
    # How far the STIM laptop's clock is AHEAD of the EEG laptop's (s),
    # from speech_lag.py.
    STIM_CLOCK_AHEAD_S = 15.6
    TMIN, TMAX = -0.2, 2.0            # epoch window around stimulus onset (s)
    BASELINE = (-0.2, 0.0)            # baseline correction window, or None
    SEGMENTS = ["word", "baseline"]   # which CSV segment types to epoch
    REJECT_UV = None                  # e.g. 150 to drop epochs > 150 µV p-p
    EXP_NAMES = {1: "exp1_mental", 2: "exp2_spoken"}
    EVENT_ID = {"baseline": 1, "word/other": 2, "word/trigger": 3}
    # ------------------------------------------------------------------------

    DATASET.mkdir(parents=True, exist_ok=True)
    report, by_exp = [], {}

    for stim_csv in sorted(PROC.glob("*/*_stim_all.csv")):
        name = stim_csv.parent.name
        qc = json.loads((stim_csv.parent / f"{name}_qc.json").read_text())
        save_dir = SAVE / name
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name}")

        raw = mne.io.read_raw_fif(stim_csv.parent / f"{name}_preproc_raw.fif",
                                  preload=True, verbose="error")
        sfreq, dur = raw.info["sfreq"], raw.times[-1]
        eeg_start = raw.info["meas_date"].timestamp()   # EEG laptop clock (UTC)

        stim = pd.read_csv(stim_csv)
        stim = stim[stim["complete"].astype(str).str.lower() == "true"]
        stim = stim[stim["segment"].isin(SEGMENTS)].reset_index(drop=True)

        # stim laptop time -> EEG laptop time -> seconds into the recording
        t_eeg = stim["onset_epoch_ms"].to_numpy() / 1000 - STIM_CLOCK_AHEAD_S - eeg_start
        last_end = t_eeg[-1] + stim["duration_ms"].iloc[-1] / 1000
        print(f"  stim runs {t_eeg[0]:.2f} -> {last_end:.2f} s   (EEG length {dur:.1f} s)")

        inside = (t_eeg + TMIN >= 0) & (t_eeg + TMAX <= dur)
        if (~inside).any():
            print(f"  ! {(~inside).sum()} of {len(t_eeg)} events fall outside the EEG, dropped")
        stim, t_eeg = stim[inside].reset_index(drop=True), t_eeg[inside]

        # labels + recording info (CSV 'trial' column = word index, kept as is)
        stim["label"] = np.where(stim["segment"] == "word",
                                 "word/" + stim["stimulus_type"].fillna("none"),
                                 "baseline")
        stim["onset_in_eeg_s"] = t_eeg
        stim["recording"] = name
        stim["experiment"] = qc["experiment"]
        stim["rec_trial"] = qc["trial"]

        unknown = set(stim["label"]) - set(EVENT_ID)
        if unknown:
            print(f"  ! unknown labels dropped: {unknown}")
            keep = stim["label"].isin(EVENT_ID).to_numpy()
            stim, t_eeg = stim[keep].reset_index(drop=True), t_eeg[keep]

        events = np.column_stack([
            np.round(t_eeg * sfreq).astype(int) + raw.first_samp,
            np.zeros(len(stim), int),
            stim["label"].map(EVENT_ID).to_numpy(),
        ])
        present = {k: v for k, v in EVENT_ID.items() if v in events[:, 2]}

        epochs = mne.Epochs(raw, events, present, tmin=TMIN, tmax=TMAX,
                            baseline=BASELINE, metadata=stim, preload=True,
                            reject=dict(eeg=REJECT_UV * 1e-6) if REJECT_UV else None,
                            reject_by_annotation=False, verbose="error")
        epochs.save(save_dir / f"{name}-epo.fif", overwrite=True, verbose="error")
        epochs.metadata.to_csv(save_dir / f"{name}_labels.csv", index=False)
        counts = {k: int(v) for k, v in epochs.metadata["label"].value_counts().items()}
        print(f"  epochs saved: {len(epochs)}  {counts}")

        rep = dict(name=name, stim_clock_ahead_s=STIM_CLOCK_AHEAD_S,
                   stim_start_in_eeg_s=round(float(t_eeg[0]), 3) if len(t_eeg) else None,
                   eeg_length_s=round(dur, 1), n_dropped=int((~inside).sum()),
                   n_epochs=len(epochs), **counts)
        (save_dir / f"{name}_alignment.json").write_text(json.dumps(rep, indent=2))
        report.append(rep)
        by_exp.setdefault(qc["experiment"], []).append(epochs)

    pd.DataFrame(report).to_csv(SAVE / "alignment_summary.csv", index=False)

    # ---- combined dataset per experiment
    print("\n================ DATASET ================")
    for exp, ep_list in sorted(by_exp.items()):
        tag = EXP_NAMES.get(exp, f"exp{exp}")
        for ep in ep_list:                       # same event_id on every file
            ep.event_id = dict(EVENT_ID)
        all_ep = mne.concatenate_epochs(ep_list, verbose="error")
        all_ep.save(DATASET / f"{tag}-epo.fif", overwrite=True, verbose="error")

        meta = all_ep.metadata.reset_index(drop=True)
        meta.to_csv(DATASET / f"{tag}_labels.csv", index=False)
        np.savez_compressed(
            DATASET / f"{tag}.npz",
            X=all_ep.get_data(picks="eeg").astype(np.float32),   # volts
            y=meta["label"].to_numpy(dtype=str),
            groups=meta["rec_trial"].to_numpy(),                 # for leave-one-trial-out CV
            words=meta["word"].fillna("").to_numpy(dtype=str),
            ch_names=np.array(all_ep.copy().pick("eeg").ch_names),
            times=all_ep.times,
            sfreq=all_ep.info["sfreq"],
        )
        counts = meta.groupby(["rec_trial", "label"]).size().unstack(fill_value=0)
        print(f"\n{tag}: {len(all_ep)} epochs, {len(all_ep.ch_names)} channels, "
              f"{len(all_ep.times)} samples ({TMIN} to {TMAX} s)")
        print(counts.to_string())
    print(f"\nSaved to {DATASET}")


if __name__ == "__main__":
    main()