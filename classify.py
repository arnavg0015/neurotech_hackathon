"""
Train and compare a few classifiers on the epoched dataset.

Reads  results/dataset/<exp>.npz  (from epoch.py), writes results/ml/:
    scores.csv        balanced accuracy for every dataset x task x model x CV
    scores.png        bar plot vs chance
    confusion.csv     confusion matrix of every run

Cross-validation:
    loto  - leave-one-trial-out: train on some recordings, test on an unseen one.
            This is the honest number.
    kfold - stratified 5-fold mixing all trials. Optimistic (train and test
            epochs come from the same recording), shown for comparison.
Cross-experiment: train on spoken (exp2), test on mental (exp1).

    pip install scikit-learn pyriemann   (pyriemann optional)
    python scripts/classify.py
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from scipy.signal import welch
from scipy.stats import binom
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "dataset"
OUT = ROOT / "results" / "ml"


def bandpower(X, sfreq, bands):
    """log mean power per channel per band -> (epochs, channels * bands)"""
    f, p = welch(X, fs=sfreq, nperseg=min(X.shape[-1], int(sfreq)), axis=-1)
    feats = [np.log(p[..., (f >= lo) & (f < hi)].mean(-1) + 1e-20) for lo, hi in bands]
    return np.stack(feats, axis=-1).reshape(len(X), -1)


def time_bins(X, sfreq, bin_s):
    """average the signal in short bins -> (epochs, channels * bins)"""
    k = max(1, int(round(bin_s * sfreq)))
    n = X.shape[-1] // k * k
    return X[..., :n].reshape(X.shape[0], X.shape[1], -1, k).mean(-1).reshape(len(X), -1)


def main():
    # ------------------------------------------------------------------ flags
    DATASETS = ["exp1_mental", "exp2_spoken"]
    TASKS = {   # original label -> class name (labels not listed are dropped)
        "trigger_vs_other": {"word/trigger": "trigger", "word/other": "other"},
        "word_vs_baseline": {"word/trigger": "word", "word/other": "word",
                             "baseline": "baseline"},
        "3class": {"word/trigger": "trigger", "word/other": "other",
                   "baseline": "baseline"},
        "trigger_vs_baseline": {"word/trigger": "trigger", "baseline": "baseline"},
    }
    FEAT_TMIN, FEAT_TMAX = 0.0, 2.0     # part of the epoch used as features (s)
    BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]   # delta..gamma
    ERP_BIN_S = 0.05                    # time-bin width for the ERP features
    CV_MODES = ["loto", "kfold"]
    KFOLD = 5
    CROSS_EXPERIMENT = True             # train exp2_spoken, test exp1_mental
    SEED = 42
    N_JOBS = -1
    # ------------------------------------------------------------------------

    mne.set_log_level("error")
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        has_riemann = True
    except ImportError:
        has_riemann = False
        print("(pyriemann not installed - skipping Riemann model)")

    # ---- load
    data = {}
    for ds in DATASETS:
        d = np.load(DATA / f"{ds}.npz")
        tmask = (d["times"] >= FEAT_TMIN) & (d["times"] <= FEAT_TMAX)
        data[ds] = dict(X=d["X"][..., tmask] * 1e6,          # volts -> µV
                        y=d["y"], groups=d["groups"], sfreq=float(d["sfreq"]))
        print(f"{ds}: X {data[ds]['X'].shape}, trials {np.unique(d['groups'])}")
    sfreq = data[DATASETS[0]]["sfreq"]

    # ---- models
    models = {
        "CSP+LDA": make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True), LDA()),
        "ERP+sLDA": make_pipeline(
            FunctionTransformer(time_bins, kw_args=dict(sfreq=sfreq, bin_s=ERP_BIN_S)),
            StandardScaler(), LDA(solver="lsqr", shrinkage="auto")),
        "Bandpower+LR": make_pipeline(
            FunctionTransformer(bandpower, kw_args=dict(sfreq=sfreq, bands=BANDS)),
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
        "Bandpower+SVM": make_pipeline(
            FunctionTransformer(bandpower, kw_args=dict(sfreq=sfreq, bands=BANDS)),
            StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
        "Bandpower+RF": make_pipeline(
            FunctionTransformer(bandpower, kw_args=dict(sfreq=sfreq, bands=BANDS)),
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=SEED)),
    }
    if has_riemann:
        models["Riemann+LR"] = make_pipeline(
            Covariances("oas"), TangentSpace(),
            LogisticRegression(max_iter=5000, class_weight="balanced"))

    # ---- build every evaluation: (eval name, task, cv, X/y for train/test)
    rows, cms = [], []
    for task, mapping in TASKS.items():
        sets = {}
        for ds, d in data.items():
            keep = np.isin(d["y"], list(mapping))
            sets[ds] = (d["X"][keep], np.array([mapping[l] for l in d["y"][keep]]),
                        d["groups"][keep])

        for ds, (X, y, g) in sets.items():
            for cv_mode in CV_MODES:
                if cv_mode == "loto" and len(np.unique(g)) < 2:
                    continue
                cv = (LeaveOneGroupOut() if cv_mode == "loto"
                      else StratifiedKFold(KFOLD, shuffle=True, random_state=SEED))
                chance = 1 / len(np.unique(y))
                thresh = binom.ppf(0.95, len(y), chance) / len(y)
                print(f"\n--- {ds} | {task} | {cv_mode}   n={len(y)}  "
                      f"chance {chance:.2f}, p<.05 above {thresh:.3f}")
                for mname, model in models.items():
                    pred = cross_val_predict(model, X, y, groups=g, cv=cv, n_jobs=N_JOBS)
                    bacc = balanced_accuracy_score(y, pred)
                    sig = "*" if bacc > thresh else ""
                    print(f"  {mname:<14} {bacc:.3f} {sig}")
                    rows.append(dict(eval=ds, task=task, cv=cv_mode, model=mname,
                                     n=len(y), balanced_acc=round(bacc, 3),
                                     chance=round(chance, 3), threshold_p05=round(thresh, 3),
                                     significant=bool(sig)))
                    labs = np.unique(y)
                    cms.append(pd.DataFrame(confusion_matrix(y, pred, labels=labs),
                                            index=[f"{ds}|{task}|{cv_mode}|{mname}|true={l}"
                                                   for l in labs],
                                            columns=[f"pred={l}" for l in labs]))

        # ---- train on spoken, test on mental
        if CROSS_EXPERIMENT and {"exp1_mental", "exp2_spoken"} <= set(sets):
            Xa, ya, _ = sets["exp2_spoken"]
            Xb, yb, _ = sets["exp1_mental"]
            chance = 1 / len(np.unique(yb))
            thresh = binom.ppf(0.95, len(yb), chance) / len(yb)
            print(f"\n--- spoken->mental | {task}   n_test={len(yb)}  "
                  f"chance {chance:.2f}, p<.05 above {thresh:.3f}")
            for mname, model in models.items():
                pred = clone(model).fit(Xa, ya).predict(Xb)
                bacc = balanced_accuracy_score(yb, pred)
                sig = "*" if bacc > thresh else ""
                print(f"  {mname:<14} {bacc:.3f} {sig}")
                rows.append(dict(eval="spoken->mental", task=task, cv="cross", model=mname,
                                 n=len(yb), balanced_acc=round(bacc, 3),
                                 chance=round(chance, 3), threshold_p05=round(thresh, 3),
                                 significant=bool(sig)))

    # ---- save
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scores.csv", index=False)
    pd.concat(cms).to_csv(OUT / "confusion.csv")

    # ---- plot: one panel per evaluation x task, bars = models, colors = cv
    evals = list(dict.fromkeys(df["eval"]))
    tasks = list(TASKS)
    cv_list = list(dict.fromkeys(df["cv"]))
    mnames = list(models)
    fig, axes = plt.subplots(len(evals), len(tasks), figsize=(5 * len(tasks), 3.6 * len(evals)),
                             squeeze=False, sharey=True)
    width = 0.8 / len(cv_list)
    for r, ev in enumerate(evals):
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            sub = df[(df["eval"] == ev) & (df["task"] == task)]
            if sub.empty:
                ax.axis("off")
                continue
            for i, cv_mode in enumerate(cv_list):
                s = sub[sub["cv"] == cv_mode].set_index("model").reindex(mnames)
                ax.bar(np.arange(len(mnames)) + i * width, s["balanced_acc"], width,
                       label=cv_mode)
            ax.axhline(sub["chance"].iloc[0], color="k", ls="--", lw=1, label="chance")
            ax.axhline(sub["threshold_p05"].iloc[0], color="tab:red", ls=":", lw=1,
                       label="p<.05")
            ax.set_xticks(np.arange(len(mnames)) + width * (len(cv_list) - 1) / 2)
            ax.set_xticklabels(mnames, rotation=40, ha="right", fontsize=8)
            ax.set_title(f"{ev} | {task}", fontsize=10)
            ax.set_ylim(0, 1)
            if c == 0:
                ax.set_ylabel("balanced accuracy")
    axes[0, -1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "scores.png", dpi=120)
    plt.close(fig)

    # ---- summary: best model per evaluation x task
    print("\n================ BEST PER TASK ================")
    best = df.loc[df.groupby(["eval", "task", "cv"])["balanced_acc"].idxmax()]
    print(best[["eval", "task", "cv", "model", "balanced_acc", "chance",
                "threshold_p05", "significant"]].to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()