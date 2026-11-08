"""
Estimate the EEG <-> stim laptop clock offset from speech muscle artifacts.

When the subject speaks a word, jaw/face muscles produce a burst of high-
frequency (30-100 Hz) activity in the EEG. This script slides the word onsets
from the CSV (actual onset times, so irregular spacing is used) across the
recording and finds the clock offset where speech bursts line up with words.

  1. Scan: for every candidate offset (e.g. -30..+30 s, 10 ms steps) compare
     EMG inside the expected speech windows vs everywhere else. The irregular
     word timing makes the correct offset a single clear peak.
  2. Refine: at the best offset, average EMG around word onsets, measure when
     the burst actually starts, and correct for ASSUMED_RT_S.

Uses the raw .cnt (the preprocessed data is low-passed at 45 Hz and ICA'd).
Plots go to results/timing/.

    python scripts/speech_lag.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_processed"
OUT = ROOT / "results" / "timing"


def main():
    # ------------------------------------------------------------------ flags
    EXPERIMENTS = [2]                 # add 1 to inspect mental trials (expect no burst)
    TRIALS_FOR_ESTIMATE = [2, 3]      # trials with clearer enunciation
    SCAN_RANGE_S = (-30.0, 30.0)      # STIM_CLOCK_AHEAD_S values to test (covers +/-16)
    SCAN_STEP_S = 0.01
    ASSUMED_RT_S = 0.5                # delay from word on screen to speech EMG onset
    SPEECH_DUR_S = 0.8                # rough length of a spoken-word burst
    EMG_CHANNELS = ["T7", "T8", "TP7", "TP8", "FT7", "FT8", "F7", "F8"]
    EMG_BAND = (30.0, 100.0)          # Hz
    LINE_FREQ = 60.0
    SMOOTH_S = 0.05                   # envelope smoothing
    ENV_SFREQ = 100                   # envelope sampling rate
    # ------------------------------------------------------------------------

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for qc_path in sorted(PROC.glob("*/*_qc.json")):
        qc = json.loads(qc_path.read_text())
        name = qc["name"]
        stim_csv = qc_path.parent / f"{name}_stim_all.csv"
        if qc.get("experiment") not in EXPERIMENTS or not stim_csv.exists():
            continue
        print(f"\n=== {name}")

        # ---- EMG envelope from the raw recording
        raw = mne.io.read_raw_ant(ROOT / qc["source_file"], preload=True, verbose="error")
        eeg_start = raw.info["meas_date"].timestamp()
        bads = set(qc.get("bad_channels", []))
        wanted = {c.upper() for c in EMG_CHANNELS}
        picks = [c for c in raw.ch_names if c.upper() in wanted and c not in bads]
        if len(picks) < 2:
            picks = [c for c in raw.ch_names if c not in bads]
        print(f"  EMG channels : {picks}")
        raw.pick(picks)

        sf = raw.info["sfreq"]
        raw.notch_filter(np.arange(LINE_FREQ, sf / 2, LINE_FREQ), verbose="error")
        raw.filter(EMG_BAND[0], min(EMG_BAND[1], sf / 2 - 5), verbose="error")
        raw.apply_hilbert(envelope=True, verbose="error")
        env = np.log10(raw.get_data().mean(axis=0) + 1e-12)

        step = max(1, int(round(sf / ENV_SFREQ)))
        fs = sf / step
        k = max(1, int(SMOOTH_S * sf))
        env = np.convolve(env, np.ones(k) / k, mode="same")[::step]
        env = (env - np.median(env)) / (env.std() + 1e-12)
        n = len(env)
        cs = np.concatenate([[0], np.cumsum(env)])

        # ---- word onsets from the CSV: stim-laptop clock, seconds after EEG start
        stim = pd.read_csv(stim_csv)
        words = stim[(stim["segment"] == "word")
                     & (stim["complete"].astype(str).str.lower() == "true")]
        on_clock = words["onset_epoch_ms"].to_numpy() / 1000 - eeg_start
        gaps = np.diff(on_clock)
        word_len = float(words["duration_ms"].median() / 1000)
        print(f"  words        : {len(on_clock)}, gap between words "
              f"{gaps.min():.2f}-{gaps.max():.2f} s (median {np.median(gaps):.2f})")

        # ---- 1. scan offsets: EMG inside speech windows minus outside
        aheads = np.arange(SCAN_RANGE_S[0], SCAN_RANGE_S[1] + SCAN_STEP_S / 2, SCAN_STEP_S)
        win = int(SPEECH_DUR_S * fs)
        scores = np.full(len(aheads), np.nan)
        for j, ah in enumerate(aheads):
            s = np.round((on_clock - ah + ASSUMED_RT_S) * fs).astype(int)
            s = s[(s >= 0) & (s + win <= n)]
            if len(s) < len(on_clock) / 2:       # need most words inside the EEG
                continue
            in_sum = (cs[s + win] - cs[s]).sum()
            in_n = len(s) * win
            scores[j] = in_sum / in_n - (cs[-1] - in_sum) / (n - in_n)

        best = int(np.nanargmax(scores))
        ahead_scan = float(aheads[best])
        # runner-up peak at least 1 s away -> how unambiguous the answer is
        far = np.abs(aheads - ahead_scan) > 1.0
        runner = int(np.nanargmax(np.where(far, scores, np.nan)))
        print(f"  scan best    : {ahead_scan:+.2f} s (score {scores[best]:.3f})   "
              f"runner-up {aheads[runner]:+.2f} s ({scores[runner]:.3f})")

        # ---- 2. refine: measure burst onset at the best offset
        t_on = on_clock - ahead_scan
        lags = np.arange(int(-1.0 * fs), int(3.0 * fs))
        tl = lags / fs
        idx = np.round(t_on * fs).astype(int)
        ok = (idx + lags[0] >= 0) & (idx + lags[-1] < n)
        mat = env[idx[ok][:, None] + lags[None, :]]
        avg = mat.mean(axis=0)

        p = int(np.argmax(avg))
        base = np.percentile(avg, 10)
        half = (base + avg[p]) / 2
        i = p
        while i > 0 and avg[i] > half:
            i -= 1
        latency = float(tl[i])
        strength = float(avg[p] - base)
        ahead_final = ahead_scan - latency + ASSUMED_RT_S
        print(f"  burst onset  : {latency:+.3f} s after word onset, strength {strength:.2f} z"
              + ("   <- WEAK, don't trust" if strength < 1 else ""))
        print(f"  final offset : STIM_CLOCK_AHEAD_S = {ahead_final:+.3f}")

        # ---- plot
        fig, ax = plt.subplots(3, 1, figsize=(9, 10))
        ax[0].plot(aheads, scores, "k", lw=0.8)
        ax[0].axvline(ahead_scan, color="tab:red", ls="--", label=f"best {ahead_scan:+.2f}s")
        ax[0].set(title=f"{name}: offset scan", xlabel="STIM_CLOCK_AHEAD_S (s)",
                  ylabel="EMG in speech windows - outside (z)")
        ax[0].legend(loc="upper right")
        ax[1].plot(tl, avg, "k")
        ax[1].axvspan(0, word_len, color="tab:blue", alpha=0.15, label="word on screen")
        ax[1].axvline(latency, color="tab:red", ls="--", label=f"burst onset {latency:+.2f}s")
        ax[1].set(title=f"EMG locked to word onset (offset {ahead_scan:+.2f}s)",
                  xlabel="time from word onset (s)", ylabel="EMG (z)")
        ax[1].legend(loc="upper right")
        ax[2].imshow(mat, aspect="auto", cmap="magma",
                     extent=[tl[0], tl[-1], len(mat), 0], vmin=-1, vmax=3)
        ax[2].axvline(0, color="w")
        ax[2].set(title="single words (each row = one word)",
                  xlabel="time from word onset (s)", ylabel="word #")
        fig.tight_layout()
        fig.savefig(OUT / f"{name}_speech_lag.png", dpi=120)
        plt.close(fig)

        rows.append(dict(name=name, trial=qc["trial"], words_used=int(ok.sum()),
                         scan_best_s=round(ahead_scan, 3),
                         runner_up_s=round(float(aheads[runner]), 3),
                         peak_ratio=round(float(scores[best] / scores[runner]), 2),
                         burst_latency_s=round(latency, 3),
                         burst_strength_z=round(strength, 2),
                         ahead_final_s=round(ahead_final, 3)))

    if not rows:
        print("No matching recordings found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "speech_lag_summary.csv", index=False)
    print("\n================ SUMMARY ================")
    print(df.to_string(index=False))

    use = df[df["trial"].isin(TRIALS_FOR_ESTIMATE) & (df["burst_strength_z"] >= 1)]
    if use.empty:
        print("\nNo trial had a clear speech burst - can't recommend an offset.")
        return
    spread = use["ahead_final_s"].max() - use["ahead_final_s"].min()
    print(f"\nRecommended STIM_CLOCK_AHEAD_S = {use['ahead_final_s'].median():+.3f} s "
          f"(from trials {list(use['trial'])}, spread {spread:.3f} s)")
    if spread > 0.2:
        print("! trials disagree by > 0.2 s - check the plots before using this")
    if (use["peak_ratio"] < 1.2).any():
        print("! scan peak barely beats the runner-up in some trials - check the plots")
    print(f"Plots: {OUT}")


if __name__ == "__main__":
    main()