"""
Shipboard Machinery Condition Monitoring — deployable dashboard.

Runs three ways, unchanged:
  * Google Colab         : demo.launch(share=True)  -> public URL
  * Local / ship LAN     : python app.py            -> http://0.0.0.0:7860
  * Hugging Face Spaces  : commit app.py + requirements.txt

It loads 'marine_cbm_model.joblib' if present. If not, it rebuilds the model
from the public UCI/ZeMA source so the app is self-bootstrapping on a bare host.
"""

import io
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import joblib
import gradio as gr

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report

RANDOM_STATE = 42
ARTIFACT_PATH = os.environ.get("CBM_ARTIFACT", "marine_cbm_model.joblib")
SOURCE_BASE = ("https://raw.githubusercontent.com/Machine-Learning-FGA/"
               "Hydraulic-systems/master/data")

STATE_NAMES = ["OPERATIONAL NORMAL", "WARNING", "CRITICAL"]
DESCRIPTORS = ["mean", "std", "min", "max", "p10", "p90", "range", "drift", "rms"]
LEVEL_DESCRIPTORS = ["mean", "min", "max", "p10", "p90", "rms"]

CHANNEL_UNITS = {
    "TS1": "degC", "TS2": "degC", "TS3": "degC", "TS4": "degC",
    "VS1": "mm/s", "FS1": "l/min", "FS2": "l/min",
    "PS1": "bar", "PS2": "bar", "EPS1": "W",
}
CHANNEL_ROLES = {
    "TS1": "Secondary circuit temperature",
    "TS2": "Circuit temperature",
    "TS3": "Circuit temperature",
    "TS4": "Cooler outlet temperature (primary indicator)",
    "VS1": "Vibration velocity RMS (ISO 20816 quantity)",
    "FS1": "Cooling / lube-oil volume flow",
    "FS2": "Cooling / lube-oil volume flow",
    "PS1": "Working / lube-oil circuit pressure",
}

ISO_VIBRATION_ZONES = [
    (2.8, "A", "Newly commissioned machinery — well within limits"),
    (7.1, "B", "Acceptable for unrestricted long-term operation"),
    (11.0, "C", "Unsatisfactory for continuous operation — short-term running only"),
    (np.inf, "D", "SEVERE — vibration sufficient to cause damage; investigate now"),
]

RECOMMENDED_ACTIONS = {
    0: ["Continue normal watchkeeping routine.",
        "Record parameters in the engine-room log at the scheduled interval.",
        "No deviation from the planned maintenance schedule."],
    1: ["Increase monitoring frequency to hourly for the affected system.",
        "Inspect the heat exchanger for fouling; check differential pressure across the cooler.",
        "Verify sea-water side flow and check the strainer for blockage.",
        "Schedule cleaning at the next port call; raise a PMS work order now.",
        "Trend the parameter over 24 h and confirm the direction of travel."],
    2: ["REDUCE LOAD on the affected machinery immediately.",
        "Bring the standby unit online and prepare for changeover.",
        "Inform the Chief Engineer; notify the bridge if propulsion or steering could be affected.",
        "Open the cooler for inspection and cleaning at the earliest safe opportunity.",
        "Do not defer — continued operation risks thermal runaway and consequential damage."],
}


# =====================================================================
# Feature engineering — identical to the notebook
# =====================================================================
def extract_cycle_features(matrix, channel):
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    half = max(x.shape[1] // 2, 1)
    return pd.DataFrame({
        f"{channel}_mean": x.mean(axis=1),
        f"{channel}_std": x.std(axis=1),
        f"{channel}_min": x.min(axis=1),
        f"{channel}_max": x.max(axis=1),
        f"{channel}_p10": np.percentile(x, 10, axis=1),
        f"{channel}_p90": np.percentile(x, 90, axis=1),
        f"{channel}_range": x.max(axis=1) - x.min(axis=1),
        f"{channel}_drift": x[:, half:].mean(axis=1) - x[:, :half].mean(axis=1),
        f"{channel}_rms": np.sqrt((x ** 2).mean(axis=1)),
    })


def _fetch(name, timeout=600):
    url = f"{SOURCE_BASE}/{name}.txt"
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (cbm-dashboard)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return pd.read_csv(io.BytesIO(payload), sep="\t", header=None,
                       dtype=np.float32).values


def build_artifact_from_source(include_pressure=True, verbose=True):
    """Rebuild the reference model from the public dataset."""
    channels = ["TS1", "TS2", "TS3", "TS4", "VS1", "FS1", "FS2"]
    blocks, used = [], []
    for channel in channels:
        blocks.append(extract_cycle_features(_fetch(channel), channel))
        used.append(channel)
    if include_pressure:
        try:
            blocks.append(extract_cycle_features(_fetch("PS1"), "PS1"))
            used.append("PS1")
        except Exception:
            pass

    X = pd.concat(blocks, axis=1)
    profile = pd.DataFrame(_fetch("profile"),
                           columns=["cooler", "valve", "pump", "accumulator", "stable"])
    y = profile["cooler"].map({100.0: 0, 20.0: 1, 3.0: 2}).astype(int).values

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                       random_state=RANDOM_STATE)),
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    macro_f1 = f1_score(y_te, model.predict(X_te), average="macro")

    normal = X.loc[y == 0]
    return {
        "mode": "descriptor",
        "model": model,
        "feature_names": list(X.columns),
        "channels": used,
        "normal_reference": X.loc[y == 0].median(),
        "bounds": pd.DataFrame({"lo": X.min(), "hi": X.max()}),
        "state_names": STATE_NAMES,
        "confidence_threshold": 0.60,
        "trained_on": "UCI/ZeMA hydraulic rig (ID 447) — reference model",
        "holdout_macro_f1": float(macro_f1),
        "n_training_cycles": int(len(y)),
        "built_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }


def load_artifact():
    if os.path.exists(ARTIFACT_PATH):
        return joblib.load(ARTIFACT_PATH)
    artifact = build_artifact_from_source()
    try:
        joblib.dump(artifact, ARTIFACT_PATH)
    except Exception:
        pass
    return artifact


ARTIFACT = load_artifact()


# =====================================================================
# Core inference
# =====================================================================
def _iso_zone(rms):
    for limit, zone, note in ISO_VIBRATION_ZONES:
        if rms <= limit:
            return zone, note
    return "D", ISO_VIBRATION_ZONES[-1][2]


def _row_from_channel_values(values, artifact):
    """Turn {channel: reading} into one model-ready feature row."""
    reference = artifact["normal_reference"].copy()
    warnings_raised = []

    for channel, value in values.items():
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue
        if artifact["mode"] == "direct":
            if channel in reference.index:
                reference[channel] = value
            continue
        for descriptor in LEVEL_DESCRIPTORS:
            column = f"{channel}_{descriptor}"
            if column in reference.index:
                reference[column] = value
        probe = f"{channel}_mean" if f"{channel}_mean" in reference.index else channel
        if probe in artifact["bounds"].index:
            lo, hi = artifact["bounds"].loc[probe, ["lo", "hi"]]
            if not (lo <= value <= hi):
                warnings_raised.append(
                    f"{channel} = {value:g} {CHANNEL_UNITS.get(channel, '')} is outside "
                    f"the training envelope [{lo:.3g}, {hi:.3g}]")
    return reference[artifact["feature_names"]].values.reshape(1, -1), warnings_raised


def diagnose_vector(vector, warnings_raised, artifact, vibration=None):
    model = artifact["model"]
    probabilities = model.predict_proba(vector)[0]
    predicted = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted])
    threshold = artifact.get("confidence_threshold", 0.60)

    sensor_fault = bool(warnings_raised)
    abstained = confidence < threshold

    if sensor_fault:
        predicted = max(predicted, 1)
        verdict = "SENSOR FAULT SUSPECTED — MANUAL VERIFICATION REQUIRED"
    elif abstained and predicted == 0:
        predicted = 1
        verdict = "WARNING — MANUAL INSPECTION REQUIRED (low confidence)"
    else:
        verdict = ["OPERATIONAL NORMAL",
                   "WARNING — DEGRADATION DETECTED",
                   "CRITICAL: MAINTENANCE REQUIRED"][predicted]

    zone = _iso_zone(vibration) if vibration is not None else (None, None)
    return {
        "verdict": verdict,
        "state_index": predicted,
        "confidence": confidence,
        "probabilities": probabilities,
        "sensor_fault": sensor_fault,
        "abstained": abstained,
        "iso_zone": zone[0],
        "iso_note": zone[1],
        "warnings": warnings_raised,
        "actions": RECOMMENDED_ACTIONS[predicted],
    }


# =====================================================================
# Gradio handlers
# =====================================================================
def default_reading_table(artifact=None):
    artifact = artifact or ARTIFACT
    rows = []
    for channel in artifact["channels"]:
        probe = (f"{channel}_mean"
                 if f"{channel}_mean" in artifact["normal_reference"].index
                 else channel)
        reference = float(artifact["normal_reference"].get(probe, np.nan))
        rows.append([channel,
                     CHANNEL_ROLES.get(channel, "user channel"),
                     CHANNEL_UNITS.get(channel, ""),
                     round(reference, 3),
                     round(reference, 3)])
    return pd.DataFrame(rows, columns=["Channel", "Description", "Unit",
                                       "Normal reference", "Your reading"])


def _verdict_banner(result):
    colour = {0: "#0f7b6c", 1: "#c98a00", 2: "#b5241c"}[result["state_index"]]
    return (
        f"<div style='padding:18px;border-radius:10px;background:{colour};"
        f"color:#fff;font-family:system-ui,sans-serif'>"
        f"<div style='font-size:12px;letter-spacing:.14em;opacity:.85'>DIAGNOSIS</div>"
        f"<div style='font-size:26px;font-weight:700;margin:6px 0'>{result['verdict']}</div>"
        f"<div style='font-size:14px;opacity:.92'>Model confidence "
        f"{result['confidence']*100:.1f}%</div></div>")


def _result_markdown(result, artifact):
    lines = []
    lines.append("### State probabilities\n")
    for name, probability in zip(artifact["state_names"], result["probabilities"]):
        bar = "█" * int(round(probability * 30))
        lines.append(f"- **{name}** — {probability*100:.2f}%  `{bar}`")

    if result["iso_zone"]:
        lines.append(f"\n### ISO 10816 / 20816 vibration zone: **{result['iso_zone']}**")
        lines.append(f"{result['iso_note']}")

    if result["warnings"]:
        lines.append("\n### ⚠ Input warnings")
        for warning in result["warnings"]:
            lines.append(f"- {warning}")
        lines.append("\nAn out-of-envelope reading is itself a fault — of the sensor. "
                     "The verdict is withheld until the transducer is verified.")

    lines.append("\n### Recommended action")
    for i, action in enumerate(result["actions"], 1):
        lines.append(f"{i}. {action}")
    return "\n".join(lines)


def handle_single(table):
    try:
        frame = pd.DataFrame(table)
        if frame.empty:
            return "", "Enter at least one reading.", None
        frame.columns = ["Channel", "Description", "Unit",
                         "Normal reference", "Your reading"][:len(frame.columns)]
        values, vibration = {}, None
        for _, row in frame.iterrows():
            channel = str(row["Channel"]).strip()
            try:
                value = float(row["Your reading"])
            except (TypeError, ValueError):
                continue
            values[channel] = value
            if channel.startswith("VS"):
                vibration = value
        if not values:
            return "", "No numeric readings found in the 'Your reading' column.", None

        vector, warnings_raised = _row_from_channel_values(values, ARTIFACT)
        result = diagnose_vector(vector, warnings_raised, ARTIFACT, vibration)

        summary = pd.DataFrame({
            "Metric": ["Verdict", "Confidence", "ISO vibration zone",
                       "Sensor fault suspected", "Channels supplied"],
            "Value": [result["verdict"], f"{result['confidence']*100:.1f}%",
                      result["iso_zone"] or "n/a",
                      "YES" if result["sensor_fault"] else "no", len(values)],
        })
        return _verdict_banner(result), _result_markdown(result, ARTIFACT), summary
    except Exception as exc:
        return "", f"**Error:** {type(exc).__name__}: {exc}", None


def _read_upload(path):
    if path is None:
        raise ValueError("No file uploaded — choose a file first.")
    name = str(path).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for sep in [",", "\t", ";"]:
        try:
            frame = pd.read_csv(path, sep=sep)
            if frame.shape[1] > 1:
                return frame
        except Exception:
            continue
    return pd.read_csv(path)


def handle_batch(path):
    try:
        frame = _read_upload(path)
        channels = [c for c in ARTIFACT["channels"] if c in frame.columns]
        if not channels:
            return (None, None,
                    f"**No recognised channel columns.** Expected any of "
                    f"`{', '.join(ARTIFACT['channels'])}`; file has "
                    f"`{', '.join(map(str, frame.columns[:12]))}`. "
                    f"Download the template below for the exact layout.")

        records = []
        for idx, row in frame.iterrows():
            values = {c: float(row[c]) for c in channels
                      if pd.notna(row[c])}
            vibration = next((v for k, v in values.items() if k.startswith("VS")), None)
            vector, warnings_raised = _row_from_channel_values(values, ARTIFACT)
            result = diagnose_vector(vector, warnings_raised, ARTIFACT, vibration)
            record = {"row": idx + 1}
            for key in ["timestamp", "unit_id", "vessel", "machine"]:
                if key in frame.columns:
                    record[key] = row[key]
            record.update({
                "verdict": result["verdict"],
                "state": ARTIFACT["state_names"][result["state_index"]],
                "confidence_%": round(result["confidence"] * 100, 2),
                "iso_zone": result["iso_zone"] or "",
                "sensor_warning": "; ".join(result["warnings"]),
            })
            records.append(record)

        results = pd.DataFrame(records)
        out_path = "cbm_batch_results.csv"
        results.to_csv(out_path, index=False)

        counts = results["state"].value_counts()
        summary = ["### Scored {:,} readings across {} channel(s)\n".format(
            len(results), len(channels))]
        for state in ARTIFACT["state_names"]:
            summary.append(f"- **{state}** — {int(counts.get(state, 0))}")
        flagged = int((results["sensor_warning"] != "").sum())
        if flagged:
            summary.append(f"\n⚠ **{flagged} reading(s)** fell outside the training "
                           f"envelope and were escalated for transducer verification.")
        summary.append(f"\nChannels used: `{', '.join(channels)}`")
        missing = [c for c in ARTIFACT["channels"] if c not in channels]
        if missing:
            summary.append(f"Channels defaulted to fleet-normal: `{', '.join(missing)}`")
        return results, out_path, "\n".join(summary)
    except Exception as exc:
        return None, None, f"**Error:** {type(exc).__name__}: {exc}"


def handle_timeseries(path, cycle_column):
    """Score raw logged time series: rows = samples, columns = channels."""
    try:
        frame = _read_upload(path)
        channels = [c for c in ARTIFACT["channels"] if c in frame.columns]
        if not channels:
            return (None, None, "**No recognised channel columns.** Expected any of "
                                f"`{', '.join(ARTIFACT['channels'])}`.")

        cycle_column = (cycle_column or "").strip()
        if cycle_column and cycle_column in frame.columns:
            groups = list(frame.groupby(cycle_column))
        else:
            groups = [("all-samples", frame)]

        records = []
        for cycle_id, block in groups:
            blocks = [extract_cycle_features(
                block[channel].to_numpy(dtype=float).reshape(1, -1), channel)
                for channel in channels]
            features = pd.concat(blocks, axis=1)

            reference = ARTIFACT["normal_reference"].copy()
            for column in features.columns:
                if column in reference.index:
                    reference[column] = features[column].iloc[0]

            warnings_raised = []
            for channel in channels:
                probe = f"{channel}_mean"
                if probe in ARTIFACT["bounds"].index:
                    lo, hi = ARTIFACT["bounds"].loc[probe, ["lo", "hi"]]
                    observed = features[probe].iloc[0]
                    if not (lo <= observed <= hi):
                        warnings_raised.append(
                            f"{channel} mean {observed:.3g} outside [{lo:.3g}, {hi:.3g}]")

            vector = reference[ARTIFACT["feature_names"]].values.reshape(1, -1)
            vibration = next((features[f"{c}_rms"].iloc[0]
                              for c in channels if c.startswith("VS")), None)
            result = diagnose_vector(vector, warnings_raised, ARTIFACT, vibration)
            records.append({
                "cycle": cycle_id,
                "samples": len(block),
                "verdict": result["verdict"],
                "state": ARTIFACT["state_names"][result["state_index"]],
                "confidence_%": round(result["confidence"] * 100, 2),
                "iso_zone": result["iso_zone"] or "",
                "sensor_warning": "; ".join(warnings_raised),
            })

        results = pd.DataFrame(records)
        out_path = "cbm_timeseries_results.csv"
        results.to_csv(out_path, index=False)
        note = (f"### Processed {len(results)} cycle(s) from {len(frame):,} samples\n"
                f"Channels used: `{', '.join(channels)}`\n\n"
                "Each cycle was reduced to the same nine descriptors "
                "(mean, std, min, max, p10, p90, range, drift, rms) used in training.")
        return results, out_path, note
    except Exception as exc:
        return None, None, f"**Error:** {type(exc).__name__}: {exc}"


def handle_retrain(path, label_column, n_trees):
    """Retrain on the vessel's own labelled data. This is the important tab."""
    global ARTIFACT
    try:
        frame = _read_upload(path)
        label_column = (label_column or "label").strip()
        if label_column not in frame.columns:
            return (f"**Label column `{label_column}` not found.** "
                    f"Columns present: `{', '.join(map(str, frame.columns[:15]))}`", None)

        label_map = {"normal": 0, "operational normal": 0, "ok": 0, "healthy": 0,
                     "warning": 1, "degraded": 1, "alert": 1,
                     "critical": 2, "failure": 2, "failed": 2}
        raw = frame[label_column]
        # NOTE: pandas 3 gives text columns a dedicated 'str' dtype rather than
        # 'object', so the old `raw.dtype == object` test silently failed and sent
        # string labels down the integer path. Test for NON-numeric instead.
        if not pd.api.types.is_numeric_dtype(raw):
            labels = raw.astype(str).str.strip().str.lower().map(label_map)
            if labels.isna().any():
                unknown = sorted(set(raw[labels.isna()].astype(str)))[:6]
                return (f"**Unrecognised label value(s):** `{unknown}`. Use 0/1/2 or "
                        f"NORMAL / WARNING / CRITICAL.", None)
            y_new = labels.astype(int).values
        else:
            y_new = raw.astype(int).values

        feature_columns = [c for c in frame.columns
                           if c != label_column
                           and pd.api.types.is_numeric_dtype(frame[c])]
        if len(feature_columns) < 1:
            return "**No numeric feature columns found.**", None
        if len(frame) < 30:
            return (f"**Only {len(frame)} rows supplied.** Retraining needs at least 30, "
                    f"and realistically several hundred labelled cycles per state.", None)

        counts = pd.Series(y_new).value_counts()
        if counts.min() < 5:
            return (f"**Too few examples in the smallest class** ({counts.min()}). "
                    f"Need at least 5; class counts: {counts.to_dict()}", None)

        X_new = frame[feature_columns]
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=int(n_trees), n_jobs=-1,
                                           random_state=RANDOM_STATE)),
        ])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_new.values, y_new, test_size=0.25, stratify=y_new,
            random_state=RANDOM_STATE)
        skf = StratifiedKFold(n_splits=min(5, int(counts.min())), shuffle=True,
                              random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="f1_macro")
        model.fit(X_tr, y_tr)
        predictions = model.predict(X_te)

        present = sorted(set(y_new))
        new_artifact = {
            "mode": "direct",
            "model": model,
            "feature_names": feature_columns,
            "channels": feature_columns,
            "normal_reference": X_new[y_new == present[0]].median(),
            "bounds": pd.DataFrame({"lo": X_new.min(), "hi": X_new.max()}),
            "state_names": STATE_NAMES,
            "confidence_threshold": 0.60,
            "trained_on": f"user-supplied vessel data ({os.path.basename(str(path))})",
            "holdout_macro_f1": float(f1_score(y_te, predictions, average="macro")),
            "n_training_cycles": int(len(frame)),
            "built_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        }
        out_path = "vessel_cbm_model.joblib"
        joblib.dump(new_artifact, out_path)
        ARTIFACT = new_artifact

        report = classification_report(
            y_te, predictions,
            labels=present,
            target_names=[STATE_NAMES[i] for i in present],
            digits=4, zero_division=0)

        message = [
            "### Retrained on your data — this model is now active",
            f"- Rows: **{len(frame):,}**   Features: **{len(feature_columns)}**",
            f"- Class counts: `{counts.sort_index().to_dict()}`",
            f"- Cross-validated macro-F1: **{cv_scores.mean():.4f}** "
            f"(± {cv_scores.std():.4f})",
            f"- Held-out macro-F1: **{new_artifact['holdout_macro_f1']:.4f}**",
            f"- Held-out accuracy: **{accuracy_score(y_te, predictions):.4f}**",
            "\n```\n" + report + "\n```",
            "\nThe **Single reading** tab has been refreshed with your channel list.",
            "\n> Treat these scores as provisional until the model has been shadow-run "
            "against the existing planned-maintenance scheme for a full survey cycle.",
        ]
        return "\n".join(message), out_path, default_reading_table(new_artifact)
    except Exception as exc:
        return f"**Error:** {type(exc).__name__}: {exc}", None, gr.update()


def make_template():
    frame = pd.DataFrame(columns=["timestamp", "unit_id"] + list(ARTIFACT["channels"]))
    reference = {}
    for channel in ARTIFACT["channels"]:
        probe = (f"{channel}_mean"
                 if f"{channel}_mean" in ARTIFACT["normal_reference"].index else channel)
        reference[channel] = round(float(ARTIFACT["normal_reference"].get(probe, 0)), 3)
    frame.loc[0] = ["2026-01-01T08:00:00", "DG-1"] + [reference[c] for c in ARTIFACT["channels"]]
    frame.loc[1] = ["2026-01-01T12:00:00", "DG-1"] + [reference[c] for c in ARTIFACT["channels"]]
    path = "cbm_input_template.csv"
    frame.to_csv(path, index=False)
    return path


def make_retrain_template():
    columns = list(ARTIFACT["channels"])
    frame = pd.DataFrame(columns=columns + ["label"])
    for i in range(3):
        reference = []
        for channel in columns:
            probe = (f"{channel}_mean"
                     if f"{channel}_mean" in ARTIFACT["normal_reference"].index else channel)
            reference.append(round(float(ARTIFACT["normal_reference"].get(probe, 0)), 3))
        frame.loc[i] = reference + [["NORMAL", "WARNING", "CRITICAL"][i]]
    path = "cbm_retrain_template.csv"
    frame.to_csv(path, index=False)
    return path


# =====================================================================
# UI
# =====================================================================
ABOUT = f"""
## Shipboard Machinery Condition Monitoring

Condition-based maintenance decision support for auxiliary cooling and hydraulic
plant, following the **ISO 13374** architecture (data acquisition → state
detection → health assessment → advisory generation).

**Active model:** {ARTIFACT['trained_on']}
**Held-out macro-F1:** {ARTIFACT.get('holdout_macro_f1', float('nan')):.4f} ·
**Training cycles:** {ARTIFACT.get('n_training_cycles', 'n/a'):,} ·
**Built:** {ARTIFACT.get('built_at', 'n/a')}

---

### ⚠ Read this before using it on a real vessel

The shipped model is trained on the **UCI/ZeMA hydraulic test rig**, whose absolute
sensor levels (cooler outlet ≈ 30–53 °C, circuit pressure ≈ 155–181 bar, vibration
≈ 0.5–0.9 mm/s) bear **no relation to those of a shipboard diesel generator or
steering-gear power pack**. Fed real ship data, it will either flag everything as
out-of-envelope or return meaningless verdicts.

That is not a defect to be worked around — it is the correct behaviour, and the
out-of-envelope warning exists precisely to catch it.

**The shipped model is a working demonstration of the method, not a deployable
diagnostic for your vessel.** To use this operationally:

1. Log your own machinery data alongside recorded condition states.
2. Use the **Retrain** tab to fit a model on that data.
3. Shadow-run it against the existing planned-maintenance scheme for a full
   survey cycle before anyone acts on its output.
4. Never let it authorise deferral of a class-mandated survey without the
   surveyor's agreement (IACS UR Z).

### Standards
ISO 13374-1/-2 · ISO 17359:2018 · ISO 10816-3/-6 · ISO 20816-1:2016 ·
ISO 20283-4 · ISO 13381-1 · ISO 8528-1 · IEEE Std 493 · SOLAS II-1 Reg. 29

### Deployment
- **Colab** — `demo.launch(share=True)` gives a public URL for ~72 h.
- **Ship LAN / local** — `python app.py`, then browse to port 7860.
- **Hugging Face Spaces** — commit `app.py` + `requirements.txt`, pick the Gradio SDK.
- **Docker** — `pip install -r requirements.txt && python app.py`.
"""

# Gradio 6 moved `theme` from the Blocks constructor to launch(). Colab may ship
# 4.x or 5.x, so route the argument by version rather than pinning a release.
_GR_MAJOR = int(str(gr.__version__).split(".")[0])
_BLOCKS_KWARGS = {"title": "Shipboard CBM Console"}
LAUNCH_KWARGS = {}
try:
    if _GR_MAJOR >= 6:
        LAUNCH_KWARGS["theme"] = gr.themes.Soft()
    else:
        _BLOCKS_KWARGS["theme"] = gr.themes.Soft()
except Exception:
    pass

with gr.Blocks(**_BLOCKS_KWARGS) as demo:
    gr.Markdown("# ⚓ Shipboard Machinery Condition Monitoring System")
    gr.Markdown("*Auxiliary cooling & hydraulic plant · ISO 13374 advisory generation*")

    with gr.Tab("1 · Single reading"):
        gr.Markdown("Edit the **Your reading** column with live values from the "
                    "engine-room gauges or data logger, then press Diagnose. "
                    "Rows left at the reference value are treated as nominal.")
        reading_table = gr.Dataframe(value=default_reading_table(),
                                     interactive=True, wrap=True,
                                     label="Sensor readings")
        diagnose_button = gr.Button("Diagnose", variant="primary")
        verdict_html = gr.HTML()
        detail_markdown = gr.Markdown()
        summary_frame = gr.Dataframe(label="Summary", interactive=False)
        diagnose_button.click(handle_single, inputs=reading_table,
                              outputs=[verdict_html, detail_markdown, summary_frame])

    with gr.Tab("2 · Batch CSV"):
        gr.Markdown("Upload a CSV/Excel file with **one row per reading** and one "
                    "column per channel. Optional `timestamp`, `unit_id`, `vessel` "
                    "columns are carried through to the results.")
        with gr.Row():
            batch_file = gr.File(label="Readings file (.csv / .xlsx)", type="filepath")
            with gr.Column():
                template_button = gr.Button("Download input template")
                template_file = gr.File(label="Template", interactive=False)
        batch_button = gr.Button("Score batch", variant="primary")
        batch_note = gr.Markdown()
        batch_table = gr.Dataframe(label="Results", interactive=False, wrap=True)
        batch_download = gr.File(label="Download results CSV", interactive=False)
        template_button.click(make_template, outputs=template_file)
        batch_button.click(handle_batch, inputs=batch_file,
                           outputs=[batch_table, batch_download, batch_note])

    with gr.Tab("3 · Raw time series"):
        gr.Markdown("Upload **raw logger output**: rows = samples, columns = channels. "
                    "This is how shipboard data actually arrives. Each duty cycle is "
                    "reduced to the same nine descriptors used in training. Give the "
                    "name of a cycle/group column to score cycles separately, or "
                    "leave blank to treat the whole file as one cycle.")
        with gr.Row():
            timeseries_file = gr.File(label="Time-series file", type="filepath")
            cycle_box = gr.Textbox(label="Cycle / group column (optional)",
                                   placeholder="e.g. cycle_id")
        timeseries_button = gr.Button("Process cycles", variant="primary")
        timeseries_note = gr.Markdown()
        timeseries_table = gr.Dataframe(label="Per-cycle results", interactive=False)
        timeseries_download = gr.File(label="Download results CSV", interactive=False)
        timeseries_button.click(handle_timeseries,
                                inputs=[timeseries_file, cycle_box],
                                outputs=[timeseries_table, timeseries_download,
                                         timeseries_note])

    with gr.Tab("4 · Retrain on vessel data"):
        gr.Markdown(
            "**This is the tab that makes the system operational.** Upload your own "
            "labelled machinery data — any numeric columns plus a `label` column "
            "containing `0/1/2` or `NORMAL/WARNING/CRITICAL` — and a fresh model is "
            "fitted, cross-validated and made active for the other tabs.")
        with gr.Row():
            retrain_file = gr.File(label="Labelled training data (.csv / .xlsx)",
                                   type="filepath")
            with gr.Column():
                label_box = gr.Textbox(label="Label column name", value="label")
                trees_slider = gr.Slider(100, 800, value=400, step=50,
                                         label="Number of trees")
                retrain_template_button = gr.Button("Download retraining template")
                retrain_template_file = gr.File(label="Template", interactive=False)
        retrain_button = gr.Button("Retrain model", variant="primary")
        retrain_report = gr.Markdown()
        retrain_download = gr.File(label="Download trained model (.joblib)",
                                   interactive=False)
        retrain_template_button.click(make_retrain_template,
                                      outputs=retrain_template_file)
        retrain_button.click(handle_retrain,
                             inputs=[retrain_file, label_box, trees_slider],
                             outputs=[retrain_report, retrain_download,
                                      reading_table])

    with gr.Tab("5 · About & deployment"):
        gr.Markdown(ABOUT)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)),
                **LAUNCH_KWARGS)
