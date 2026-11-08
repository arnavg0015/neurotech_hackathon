"""
Preprocess ANT Neuro .cnt EEG recordings for the Neurotech Hackathon.

Put this file in  D:\\Work\\Neurotech_Hackathon\\scripts\\  and run:
    pip install "mne>=1.9" antio pandas matplotlib scikit-learn
    python scripts/preprocess_eeg.py

Output: data_processed/<subj>_exp<N>_trial<N>/  (baseline -> <subj>_baseline/)
    <name>_preproc_raw.fif     cleaned continuous EEG
    <name>_qc.json             bad channels, ICA components removed, etc.
    <name>_stim*.json/.csv     stimulation/event files, copied + renamed
data_processed/summary.csv     one row per recording
"""
import json
import re
import shutil
from pathlib import Path

import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_EEG = ROOT / "data" / "EEG Data"
RAW_STIM = ROOT / "data" / "Stimulation Data"
OUT = ROOT / "data_processed"


def main():
    # ------------------------------------------------------------------ flags
    SUBJECT_IDS = {"Yuan_Tina": "sub01"}   # raw filename name -> subject ID
    DROP_CHANNELS = ["M1", "M2"]           # removed from all recordings
    L_FREQ = 0.5          # high-pass (Hz)
    H_FREQ = 125.0         # low-pass (Hz)
    LINE_FREQ = 60.0      # mains noise (US)
    SFREQ = 500.0         # resample target (Hz)
    BAD_Z = 10.0           # robust z-score threshold for noisy channels
    FLAT_SD = 1e-7        # channels with SD below this (V) are flat
    INTERPOLATE_BADS = True
    RUN_ICA = True
    ICA_MAX_COMPS = 20
    ICA_MAX_EXCLUDE = 3
    # ------------------------------------------------------------------------

    files = sorted(RAW_EEG.rglob("*.cnt"))
    print(f"Found {len(files)} .cnt files")
    summary = []

    for cnt in files:
        # ---- name the output: sub01_exp1_trial1 / sub01_baseline
        m = re.match(r"^(.+?)_\d{4}-\d{2}-\d{2}", cnt.stem)
        subj = SUBJECT_IDS.get(m.group(1), m.group(1).lower()) if m else "unknown"
        folders = cnt.relative_to(RAW_EEG).parts
        if any(p.lower() == "baseline" for p in folders):
            exp, trial = 0, 1
            name = f"{subj}_baseline"
            stim_dir = RAW_STIM / "Baseline"
        else:
            exp = int(re.search(r"Experiment\s*(\d+)", str(cnt), re.I).group(1))
            trial = int(re.search(r"Trial\s*(\d+)", str(cnt), re.I).group(1))
            name = f"{subj}_exp{exp}_trial{trial}"
            stim_dir = RAW_STIM / f"Experiment {exp}" / f"Trial {trial}"

        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {cnt.name}  ->  {name}")

        try:
            # ---- load + montage (positions needed for interpolation)
            raw = mne.io.read_raw_ant(cnt, preload=True, verbose="error")
            drop = [c for c in raw.ch_names if c.upper() in {d.upper() for d in DROP_CHANNELS}]
            raw.drop_channels(drop)
            print(f"  dropped: {drop}")
            raw.set_montage("colin27_1005", match_case=False, on_missing="warn")
            orig_sfreq = raw.info["sfreq"]

            # ---- filter + resample
            notch = np.arange(LINE_FREQ, raw.info["sfreq"] / 2, LINE_FREQ)
            raw.notch_filter(notch, verbose="error")
            raw.filter(L_FREQ, H_FREQ, verbose="error")
            if raw.info["sfreq"] > SFREQ:
                raw.resample(SFREQ, verbose="error")

            # ---- bad channels: flat, or extreme variance (robust z on log SD)
            sd = raw.get_data().std(axis=1)
            log_sd = np.log(sd + 1e-20)
            med = np.median(log_sd)
            mad = np.median(np.abs(log_sd - med)) * 1.4826 + 1e-12
            z = (log_sd - med) / mad

            bads = []
            for ch, s, zz in zip(raw.ch_names, sd, z):
                if s < FLAT_SD:
                    bads.append(ch)
                    print(f"  BAD  {ch:<6} flat   (SD = {s * 1e6:.3f} µV)")
                elif abs(zz) > BAD_Z:
                    bads.append(ch)
                    kind = "noisy" if zz > 0 else "low"
                    print(f"  BAD  {ch:<6} {kind:<6} (SD = {s * 1e6:.1f} µV, z = {zz:+.1f})")
            if not bads:
                print("  no bad channels")

            raw.info["bads"] = bads
            if bads and INTERPOLATE_BADS:
                raw.interpolate_bads(reset_bads=True, verbose="error")

            raw.set_eeg_reference("average", projection=False, verbose="error")

            # ---- ICA: remove eye-blink components (Fp1/Fp2 as EOG proxy)
            ica_excluded = []
            if RUN_ICA:
                rank = mne.compute_rank(raw, rank=None, verbose="error")["eeg"]
                n_comp = int(min(ICA_MAX_COMPS, rank - 1))
                fit_raw = raw.copy().filter(1.0, None, verbose="error")
                ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica",
                                            rng=97, max_iter="auto")
                ica.fit(fit_raw, verbose="error")
                for ch in [c for c in ("Fp1", "Fp2", "Fpz") if c in raw.ch_names]:
                    idx, _ = ica.find_bads_eog(fit_raw, ch_name=ch, verbose="error")
                    ica_excluded += [i for i in idx if i not in ica_excluded]
                ica.exclude = [int(i) for i in ica_excluded[:ICA_MAX_EXCLUDE]]
                ica_excluded = ica.exclude
                if ica.exclude:
                    ica.apply(raw, verbose="error")
                ica.save(out_dir / f"{name}_ica.fif", overwrite=True, verbose="error")
                print(f"  ICA removed components: {ica_excluded}")

            raw.save(out_dir / f"{name}_preproc_raw.fif", overwrite=True, verbose="error")

            # ---- copy stim/event CSVs + JSON with matching names
            #      stim_<timestamp>_triggers.csv -> <name>_stim_triggers.csv
            stim_files = []
            for f in sorted(stim_dir.glob("*")):
                ts = re.search(r"\d{4}-\d{2}-\d{2}T[\d-]+Z", f.name)
                if ts:
                    dst = out_dir / f"{name}_stim{f.name[ts.end():]}"
                    shutil.copy2(f, dst)
                    stim_files.append(dst.name)

            qc = dict(name=name, subject=subj, experiment=exp, trial=trial,
                      source_file=str(cnt.relative_to(ROOT)),
                      orig_sfreq=orig_sfreq, sfreq=raw.info["sfreq"],
                      duration_s=round(raw.times[-1], 2),
                      n_channels=len(raw.ch_names), dropped_channels=drop,
                      bad_channels=bads, n_bad=len(bads),
                      interpolated=INTERPOLATE_BADS and bool(bads),
                      ica_components_removed=ica_excluded,
                      stim_files=stim_files)
            (out_dir / f"{name}_qc.json").write_text(json.dumps(qc, indent=2))
            summary.append(qc)

        except Exception as e:
            print(f"  !! FAILED: {e}")
            summary.append(dict(name=name, source_file=str(cnt), error=str(e)))

    # ---- summary
    df = pd.DataFrame(summary)
    df.to_csv(OUT / "summary.csv", index=False)
    print("\n================ BAD CHANNEL SUMMARY ================")
    for row in summary:
        print(f"  {row['name']:<20} {', '.join(row.get('bad_channels', [])) or '-'}")
    print(f"\nDone. Summary saved to {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()