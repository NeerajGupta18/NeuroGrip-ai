"""
train.py - builds everything the API serves, from the raw dataset.

    python -m training.train --raw /path/to/unzipped/dataset [--baselines]

Input : the UCI "sEMG for Basic Hand movements" dataset (CC BY 4.0),
        https://archive.ics.uci.edu/dataset/313  - unzip it and point --raw at
        the folder containing "Database 1" and "Database 2".

Output: app/artifacts/model.json        population models + safety thresholds
        app/artifacts/evaluation.json   every number quoted in the README
        data/recordings.npz             the recordings the dashboard replays
        ../docs/EVALUATION.md           the evaluation written up

Honesty rules this script follows:
  * Windows of one recording never sit on both sides of a split.
  * "Unseen wearer" numbers come from leave-one-subject-out: the model and the
    safety thresholds for wearer X are built from the other four people only.
  * Calibration is scored only on recordings that were not used to calibrate.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import scipy.io as sio

from app.anomaly import BLOCK, channel_stats, detect_anomalies
from app.faults import FAULT_TYPES, inject_fault
from app.features import FEATURE_NAMES, SAMPLE_RATE, STEP, WINDOW, extract_features
from app.model import (GRASPS, GRASP_LABELS, PRIOR_WINDOWS, class_means, personalised_means,
                       pooled_covariance, softmax, window_scores)

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
ARTIFACTS = os.path.join(BACKEND, "app", "artifacts")
DATA_DIR = os.path.join(BACKEND, "data")
DOCS = os.path.join(os.path.dirname(BACKEND), "docs")

SOURCE_FILES = {  # dashboard id -> dataset file
    "wearer_a": "female_1", "wearer_b": "female_2", "wearer_c": "female_3",
    "wearer_d": "male_1", "wearer_e": "male_2",
}
CALIBRATION_REPS = [0, 1, 2, 3, 5, 10]
CONFIDENCE_FLOORS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_mat(path: str) -> np.ndarray:
    """-> (6 grasps, trials, T, 2 channels)"""
    m = sio.loadmat(path)
    return np.stack([np.stack([m[f"{g}_ch1"], m[f"{g}_ch2"]], axis=-1) for g in GRASPS])


def featurise(X: np.ndarray):
    """-> features (n, F), grasp label per window, recording id per window."""
    feats, labels, rec_ids = [], [], []
    for g in range(X.shape[0]):
        for t in range(X.shape[1]):
            f = extract_features(X[g, t])
            feats.append(f)
            labels += [g] * len(f)
            rec_ids += [g * 1000 + t] * len(f)
    return np.concatenate(feats), np.array(labels), np.array(rec_ids)


def fit(F, y):
    return class_means(F, y), np.linalg.inv(pooled_covariance(F, y))


def recording_scores(F, rec, means, precision):
    """Mean window score per recording -> (ids, (n_rec, C) scores)."""
    s = window_scores(F, means, precision)
    ids = np.unique(rec)
    return ids, np.stack([s[rec == i].mean(axis=0) for i in ids]), s


def accuracy(F, y, rec, means, precision):
    ids, rs, s = recording_scores(F, rec, means, precision)
    truth = ids // 1000
    return float((s.argmax(1) == y).mean()), float((rs.argmax(1) == truth).mean())


def pct(x):
    return round(100 * float(x), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--baselines", action="store_true", help="also benchmark scikit-learn models")
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    raw = {sid: load_mat(os.path.join(args.raw, "Database 1", f"{name}.mat"))
           for sid, name in SOURCE_FILES.items()}
    data = {sid: featurise(X) for sid, X in raw.items()}
    sids = list(raw)
    ev = {"dataset": {
        "name": "sEMG for Basic Hand movements (UCI id 313)", "license": "CC BY 4.0",
        "wearers": len(sids), "grasps": len(GRASPS), "recordings_per_grasp": int(raw[sids[0]].shape[1]),
        "recording_seconds": raw[sids[0]].shape[2] / SAMPLE_RATE, "sample_rate_hz": SAMPLE_RATE,
        "channels": 2, "window_seconds": WINDOW / SAMPLE_RATE, "step_seconds": STEP / SAMPLE_RATE,
        "windows_per_recording": int(len(extract_features(raw[sids[0]][0, 0]))),
        "chance_accuracy": pct(1 / len(GRASPS)),
    }}

    # ---------- 1. within-wearer (5-fold over recordings) ----------
    within, confusion = {}, np.zeros((6, 6), int)
    for sid, (F, y, rec) in data.items():
        w_acc, r_acc = [], []
        for k in range(5):
            test = (rec % 1000) % 5 == k
            means, P = fit(F[~test], y[~test])
            a, b = accuracy(F[test], y[test], rec[test], means, P)
            w_acc.append(a); r_acc.append(b)
            ids, rs, _ = recording_scores(F[test], rec[test], means, P)
            for truth, pred in zip(ids // 1000, rs.argmax(1)):
                confusion[truth, pred] += 1
        within[sid] = {"window": pct(np.mean(w_acc)), "recording": pct(np.mean(r_acc))}
    ev["within_wearer"] = {
        "per_wearer": within,
        "window": round(np.mean([v["window"] for v in within.values()]), 1),
        "recording": round(np.mean([v["recording"] for v in within.values()]), 1),
        "confusion_matrix": {"grasps": GRASPS, "rows_true_cols_predicted": confusion.tolist()},
    }

    # ---------- 2. population models (leave-one-wearer-out) ----------
    population = {}
    for sid in sids:
        F = np.concatenate([data[o][0] for o in sids if o != sid])
        y = np.concatenate([data[o][1] for o in sids if o != sid])
        population[sid] = fit(F, y)

    unseen = {}
    for sid, (F, y, rec) in data.items():
        a, b = accuracy(F, y, rec, *population[sid])
        unseen[sid] = {"window": pct(a), "recording": pct(b)}
    ev["unseen_wearer_no_calibration"] = {
        "per_wearer": unseen,
        "window": round(np.mean([v["window"] for v in unseen.values()]), 1),
        "recording": round(np.mean([v["recording"] for v in unseen.values()]), 1),
    }

    # ---------- 3. calibration curve ----------
    def calibrated_accuracy(sid, reps, prior_windows, draws=20, collect=None):
        F, y, rec = data[sid]
        pop_means, P = population[sid]
        trial = rec % 1000
        n_trials = trial.max() + 1
        out = []
        for _ in range(draws):
            chosen = rng.permutation(n_trials)[:reps]
            cal = np.isin(trial, chosen)
            sums = np.stack([F[cal & (y == c)].sum(axis=0) for c in range(6)])
            counts = np.array([(cal & (y == c)).sum() for c in range(6)])
            means = personalised_means(pop_means, sums, counts, prior_windows)
            test = ~cal
            out.append(accuracy(F[test], y[test], rec[test], means, P)[1])
            if collect is not None:
                ids, rs, _ = recording_scores(F[test], rec[test], means, P)
                collect.append((rs, ids // 1000))
        return float(np.mean(out))

    curve = {}
    for reps in CALIBRATION_REPS:
        per = {sid: pct(calibrated_accuracy(sid, reps, PRIOR_WINDOWS)) for sid in sids}
        curve[str(reps)] = {"per_wearer": per, "recording": round(np.mean(list(per.values())), 1)}
    ev["calibration_curve"] = {"prior_windows": PRIOR_WINDOWS, "reps_per_grasp": curve}

    prior_study = {}
    for prior in [0.5, PRIOR_WINDOWS, 50.0, 250.0]:
        prior_study[str(prior)] = {
            str(reps): round(np.mean([pct(calibrated_accuracy(s, reps, prior, draws=8)) for s in sids]), 1)
            for reps in [1, 3, 10]
        }
    ev["prior_strength_study"] = prior_study

    # ---------- 4. confidence scale, reliability and abstention (after 3 calibration reps) ----------
    held_out = {}
    for sid in sids:
        collected = []
        calibrated_accuracy(sid, 3, PRIOR_WINDOWS, draws=20, collect=collected)
        held_out[sid] = (np.concatenate([r for r, _ in collected]), np.concatenate([t for _, t in collected]))

    def nll(scale, rs, truth):
        return -np.log(softmax(scale * rs)[np.arange(len(truth)), truth] + 1e-12).mean()

    def best_scale(train_sids):
        rs = np.concatenate([held_out[o][0] for o in train_sids]); truth = np.concatenate([held_out[o][1] for o in train_sids])
        grid = np.arange(1.0, 23.01, 0.25)
        return float(grid[np.argmin([nll(g, rs, truth) for g in grid])])

    evidence_scale = {sid: best_scale([o for o in sids if o != sid]) for sid in sids}
    conf, correct = [], []
    for sid in sids:                      # each wearer scored with a scale fitted on the other four
        rs, truth = held_out[sid]
        probs = softmax(evidence_scale[sid] * rs)
        conf.append(probs.max(1)); correct.append(probs.argmax(1) == truth)
    conf = np.concatenate(conf); correct = np.concatenate(correct)
    abstention = []
    for floor in CONFIDENCE_FLOORS:
        keep = conf >= floor
        abstention.append({"confidence_floor": floor, "acted_on": pct(keep.mean()),
                           "accuracy_when_acting": pct(correct[keep].mean())})
    bins = []
    for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, .99), (.99, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            bins.append({"confidence_range": [lo, min(hi, 1.0)], "share": pct(m.mean()),
                         "mean_confidence": pct(conf[m].mean()), "accuracy": pct(correct[m].mean())})
    ev["confidence"] = {"after_reps": 3, "evidence_scale": evidence_scale, "abstention": abstention,
                        "reliability": bins}

    # ---------- 5. safety gate ----------
    def thresholds_from(train_sids):
        flat, peak, hum = [], [], []
        for o in train_sids:
            for recording in raw[o].reshape(-1, raw[o].shape[2], 2):
                for ch in range(2):
                    st = channel_stats(recording[:, ch])
                    flat.append(st["block_std"].min()); peak.append(st["peak"]); hum.append(st["hum_fraction"])
        return {
            "flat_std": round(0.5 * float(np.min(flat)), 5),           # half the quietest clean block
            "peak": round(1.25 * float(np.max(peak)), 3),               # 25% above the largest clean peak
            "hum_fraction": round(min(0.9, 1.15 * float(np.max(hum))), 4),
        }

    thresholds = {sid: thresholds_from([o for o in sids if o != sid]) for sid in sids}
    clean_total = false_alarms = 0
    detected = {f: [0, 0] for f in FAULT_TYPES}
    for sid in sids:
        for recording in raw[sid].reshape(-1, raw[sid].shape[2], 2):
            clean_total += 1
            false_alarms += bool(detect_anomalies(recording, thresholds[sid]))
            for fault in FAULT_TYPES:
                corrupted = inject_fault(recording, fault, rng)
                detected[fault][1] += 1
                detected[fault][0] += any(a["type"] == fault for a in detect_anomalies(corrupted, thresholds[sid]))
    ev["safety_gate"] = {
        "clean_recordings": clean_total,
        "false_alarm_rate": pct(false_alarms / clean_total),
        "false_alarms": int(false_alarms),
        "injected_fault_detection": {f: pct(h / n) for f, (h, n) in detected.items()},
        "note": "Thresholds for each wearer are set from the other four wearers only. "
                "Faults are synthetic; detection rates are a sanity check, the false-alarm rate is the real test.",
    }

    # ---------- 6. day-to-day drift (Database 2: one person, three days) ----------
    days = {}
    for path in sorted(glob.glob(os.path.join(args.raw, "Database 2", "*.mat"))):
        days[os.path.basename(path)[:-4]] = featurise(load_mat(path))
    if len(days) == 3:
        F1, y1, r1 = days["male_day_1"]
        train = (r1 % 1000) < 70
        m_same, P_same = fit(F1[train], y1[train])
        m1, P1 = fit(F1, y1)
        drift = {"same_day": pct(accuracy(F1[~train], y1[~train], r1[~train], m_same, P_same)[1])}
        for day in ["male_day_2", "male_day_3"]:
            F, y, rec = days[day]
            entry = {"no_recalibration": pct(accuracy(F, y, rec, m1, P1)[1])}
            for reps in [3, 10]:
                cal = (rec % 1000) < reps
                sums = np.stack([F[cal & (y == c)].sum(axis=0) for c in range(6)])
                counts = np.array([(cal & (y == c)).sum() for c in range(6)])
                means = personalised_means(m1, sums, counts)
                entry[f"recalibrated_{reps}_reps"] = pct(accuracy(F[~cal], y[~cal], rec[~cal], means, P1)[1])
            drift[day] = entry
        ev["day_to_day_drift"] = drift

    # ---------- 7. baselines (optional, needs scikit-learn) ----------
    if args.baselines:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        makers = {
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
            "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=3)),
            "Random forest": lambda: RandomForestClassifier(200, n_jobs=-1, random_state=0),
        }
        base = {}
        for name, make in makers.items():
            w_in, w_cross = [], []
            for sid, (F, y, rec) in data.items():
                for k in range(5):
                    test = (rec % 1000) % 5 == k
                    w_in.append((make().fit(F[~test], y[~test]).predict(F[test]) == y[test]).mean())
                Ftr = np.concatenate([data[o][0] for o in sids if o != sid])
                ytr = np.concatenate([data[o][1] for o in sids if o != sid])
                w_cross.append((make().fit(Ftr, ytr).predict(F) == y).mean())
            base[name] = {"within_wearer_window": pct(np.mean(w_in)), "unseen_wearer_window": pct(np.mean(w_cross))}
        base["Gaussian shared-covariance (deployed)"] = {
            "within_wearer_window": ev["within_wearer"]["window"],
            "unseen_wearer_window": ev["unseen_wearer_no_calibration"]["window"],
        }
        ev["baselines"] = base

    # ---------- 8. latency ----------
    sample = raw[sids[0]][0, 0]
    means, P = population[sids[0]]
    t0 = time.perf_counter()
    for _ in range(200):
        window_scores(extract_features(sample), means, P).mean(axis=0)
    ev["latency_ms_per_recording"] = round(1000 * (time.perf_counter() - t0) / 200, 2)

    # ---------- write artifacts ----------
    os.makedirs(ARTIFACTS, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(DOCS, exist_ok=True)
    model = {
        "feature_names": FEATURE_NAMES, "grasps": GRASPS,
        "window": WINDOW, "step": STEP, "sample_rate": SAMPLE_RATE,
        "sources": {
            sid: {
                "trained_on": [o for o in sids if o != sid],
                "population_means": np.round(population[sid][0], 6).tolist(),
                "precision": np.round(population[sid][1], 6).tolist(),
                "thresholds": thresholds[sid],
                "evidence_scale": evidence_scale[sid],
            } for sid in sids
        },
    }
    with open(os.path.join(ARTIFACTS, "model.json"), "w") as f:
        json.dump(model, f)
    with open(os.path.join(ARTIFACTS, "evaluation.json"), "w") as f:
        json.dump(ev, f, indent=2)

    # Recordings for the dashboard, stored as int16 to keep the repo small.
    scale = max(float(np.abs(X).max()) for X in raw.values()) / 32767
    np.savez_compressed(
        os.path.join(DATA_DIR, "recordings.npz"), scale=np.float64(scale),
        **{sid: np.round(X / scale).astype(np.int16) for sid, X in raw.items()},
    )

    write_report(ev)
    print(json.dumps({k: ev[k] for k in ["within_wearer", "unseen_wearer_no_calibration", "safety_gate",
                                         "latency_ms_per_recording"]}, indent=1)[:1500])
    print("calibration:", {k: v["recording"] for k, v in ev["calibration_curve"]["reps_per_grasp"].items()})


def write_report(ev):
    d = ev["dataset"]
    L = []
    L += ["# NeuroGrip AI - evaluation", "",
          "Generated by `backend/training/train.py`. Every number in the README comes from this file.", "",
          "## Data", "",
          f"{d['name']}, {d['license']}. {d['wearers']} healthy people each performed {d['grasps']} everyday grasps "
          f"{d['recordings_per_grasp']} times; each recording is {d['recording_seconds']:.0f} s of {d['channels']}-channel "
          f"surface EMG at {d['sample_rate_hz']} Hz. Recordings are cut into {d['window_seconds']} s windows every "
          f"{d['step_seconds']} s ({d['windows_per_recording']} per recording). Chance is {d['chance_accuracy']}%.", "",
          "## 1. A model trained on the wearer themselves", "",
          "5-fold cross-validation over recordings, so no recording is split between train and test.", "",
          "| Wearer | Single 0.5 s window | Whole recording |", "|---|---|---|"]
    for sid, v in ev["within_wearer"]["per_wearer"].items():
        L.append(f"| {sid} | {v['window']}% | {v['recording']}% |")
    L += [f"| **mean** | **{ev['within_wearer']['window']}%** | **{ev['within_wearer']['recording']}%** |", "",
          "Confusion matrix (whole recordings, all wearers pooled; rows = true grasp):", "",
          "| | " + " | ".join(GRASP_LABELS[g] for g in GRASPS) + " |", "|---|" + "---|" * len(GRASPS)]
    for g, row in zip(GRASPS, ev["within_wearer"]["confusion_matrix"]["rows_true_cols_predicted"]):
        L.append(f"| **{GRASP_LABELS[g]}** | " + " | ".join(str(n) for n in row) + " |")
    u = ev["unseen_wearer_no_calibration"]
    L += ["", "## 2. The same model on a person it has never seen", "",
          "Leave-one-wearer-out: train on four people, test on the fifth, no calibration.", "",
          "| Wearer | Single window | Whole recording |", "|---|---|---|"]
    for sid, v in u["per_wearer"].items():
        L.append(f"| {sid} | {v['window']}% | {v['recording']}% |")
    L += [f"| **mean** | **{u['window']}%** | **{u['recording']}%** |", "",
          f"This is the central finding: accuracy collapses from {ev['within_wearer']['recording']}% to "
          f"{u['recording']}%, barely above the {d['chance_accuracy']}% chance level. With two electrodes, where each "
          "grasp sits in feature space depends on the person's anatomy and electrode placement. "
          "A bionic hand cannot ship with one fixed model; it has to be calibrated to its wearer.", "",
          "## 3. Calibration recovers it", "",
          "The wearer performs each grasp a few times. Only the class means are re-estimated "
          "(`personalised_means` in `model.py`); the covariance learned from other people is kept. "
          "Scored on that wearer's remaining recordings, averaged over 20 random choices of calibration recordings.", "",
          "| Calibration recordings per grasp | Whole-recording accuracy |", "|---|---|"]
    for reps, v in ev["calibration_curve"]["reps_per_grasp"].items():
        L.append(f"| {reps} | {v['recording']}% |")
    L += ["", "How strongly should other people's class means be trusted? "
          "Prior strength is measured in windows; one recording is "
          f"{d['windows_per_recording']} windows.", "",
          "| Prior strength (windows) | 1 rep | 3 reps | 10 reps |", "|---|---|---|---|"]
    for prior, row in ev["prior_strength_study"].items():
        mark = " (deployed)" if float(prior) == ev["calibration_curve"]["prior_windows"] else ""
        L.append(f"| {prior}{mark} | {row['1']}% | {row['3']}% | {row['10']}% |")
    L += ["", "A weak prior wins: what transfers between people is the covariance (the distance metric), "
          "not the class means.", ""]
    if "day_to_day_drift" in ev:
        dr = ev["day_to_day_drift"]
        L += ["## 4. Day-to-day drift", "",
              "Database 2: one person recorded on three consecutive days (electrodes re-applied each day). "
              "Model trained on day 1.", "",
              "| Test day | No recalibration | 3 reps/grasp | 10 reps/grasp |", "|---|---|---|---|",
              f"| Day 1 (held-out recordings) | {dr['same_day']}% | - | - |"]
        for day in ["male_day_2", "male_day_3"]:
            e = dr[day]
            L.append(f"| Day {day[-1]} | {e['no_recalibration']}% | {e['recalibrated_3_reps']}% | {e['recalibrated_10_reps']}% |")
        L += ["", "Re-applying the electrodes shifts the signal enough to roughly halve accuracy, and a short "
              "recalibration only partly restores it. This person's recordings are also harder than Database 1 "
              "even on the same day. Daily drift is an open problem for this approach, not a solved one.", ""]
    c = ev["confidence"]
    L += ["## 5. Confidence and abstention", "",
          f"After {c['after_reps']} calibration recordings per grasp. Window scores are averaged and multiplied by one "
          "fitted number, the evidence scale (see `predict` in `model.py`), before the softmax. For each wearer it is "
          f"fitted on the other four: {', '.join(f'{k} {v}' for k, v in c['evidence_scale'].items())} "
          f"(a recording has {d['windows_per_recording']} overlapping windows). The agent refuses to act below a confidence floor.", "",
          "| Confidence floor | Recordings acted on | Accuracy when acting |", "|---|---|---|"]
    for row in c["abstention"]:
        L.append(f"| {row['confidence_floor']} | {row['acted_on']}% | {row['accuracy_when_acting']}% |")
    L += ["", "Is the confidence honest? (reported confidence vs how often it was right)", "",
          "| Confidence range | Share of predictions | Mean confidence | Actual accuracy |", "|---|---|---|---|"]
    for b in c["reliability"]:
        L.append(f"| {b['confidence_range'][0]}-{b['confidence_range'][1]} | {b['share']}% | {b['mean_confidence']}% | {b['accuracy']}% |")
    s = ev["safety_gate"]
    L += ["", "## 6. Safety gate", "",
          f"False alarms on real clean recordings: **{s['false_alarms']} of {s['clean_recordings']} "
          f"({s['false_alarm_rate']}%)**, with thresholds set from the other four wearers.", "",
          "| Injected fault | Detected |", "|---|---|"]
    for f, v in s["injected_fault_detection"].items():
        L.append(f"| {f} | {v}% |")
    L += ["", s["note"], ""]
    if "baselines" in ev:
        L += ["## 7. Why not a bigger model?", "",
              "Single-window accuracy, same splits.", "",
              "| Model | Trained on the wearer | Unseen wearer |", "|---|---|---|"]
        for name, v in ev["baselines"].items():
            L.append(f"| {name} | {v['within_wearer_window']}% | {v['unseen_wearer_window']}% |")
        L += ["", "More flexible models gain a few points within a wearer and nothing on an unseen wearer. "
              "The bottleneck is the difference between people, not model capacity, so the deployed model is the "
              "one that can be personalised from a handful of recordings and explained exactly.", ""]
    L += ["## Latency", "", f"{ev['latency_ms_per_recording']} ms to featurise and classify one 6 s recording "
          "(single CPU core, NumPy).", "",
          "## Limits", "",
          "- Five healthy wearers, two electrodes, six grasps. Small by any standard.",
          "- Recordings are replayed, not streamed from hardware. No amputee data.",
          "- Calibration here uses correctly labelled recordings; a real fitting session has label noise.",
          "- Sensor faults are synthetic.", ""]
    with open(os.path.join(DOCS, "EVALUATION.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
