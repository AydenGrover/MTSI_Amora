"""
Feature Extraction Script
-------------------------
Takes raw per-frame LD2450 logs (from collect_data.py) and turns them into
windowed features suitable for a classifier.

Usage:
    python extract_features.py --input data/ --window 5 --out features.csv

    --input     Folder containing your raw session CSVs (one per recording)
    --window    Window size in seconds (e.g. 5 = 5-second windows)
    --out       Output features CSV, one row per window

Optionally define your own zones below (ZONES dict) matching your room
layout -- e.g. bed area vs desk area -- since zone dwell time is one of
the most useful features for telling sleeping apart from studying.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# Define rough rectangular zones in mm, relative to the sensor's coordinate
# origin (sensor is at x=0,y=0, y is "distance away from sensor").
# EDIT THESE to match your actual room setup before running.
ZONES = {
    "bed":  {"x_min": -2000, "x_max": 0,    "y_min": 500, "y_max": 3000},
    "desk": {"x_min": 0,     "x_max": 2000, "y_min": 500, "y_max": 3000},
}


def in_zone(x, y, zone):
    return zone["x_min"] <= x <= zone["x_max"] and zone["y_min"] <= y <= zone["y_max"]


def load_all_sessions(input_dir):
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["session_file"] = os.path.basename(f)
        dfs.append(df)
    if not dfs:
        raise SystemExit(f"No CSV files found in {input_dir}")
    return pd.concat(dfs, ignore_index=True)


def primary_target_present(row):
    """Return the (x, y, speed) of whichever target slot has a nonzero
    reading, preferring target 1. Assumes single-person recordings."""
    for i in (1, 2, 3):
        x, y, speed = row[f"t{i}_x"], row[f"t{i}_y"], row[f"t{i}_speed"]
        if x != 0 or y != 0:
            return x, y, speed
    return 0, 0, 0


def extract_window_features(window_df):
    xs, ys, speeds, present_flags = [], [], [], []
    zone_counts = {z: 0 for z in ZONES}

    for _, row in window_df.iterrows():
        x, y, speed = primary_target_present(row)
        present = 1 if (x != 0 or y != 0) else 0
        present_flags.append(present)
        if present:
            xs.append(x)
            ys.append(y)
            speeds.append(speed)
            for zname, zbounds in ZONES.items():
                if in_zone(x, y, zbounds):
                    zone_counts[zname] += 1

    n = len(window_df)
    detection_rate = float(np.mean(present_flags)) if n else 0.0

    if xs:
        feats = {
            "mean_x": np.mean(xs), "var_x": np.var(xs),
            "mean_y": np.mean(ys), "var_y": np.var(ys),
            "mean_speed": np.mean(np.abs(speeds)),
            "var_speed": np.var(speeds),
            "max_speed": np.max(np.abs(speeds)),
        }
    else:
        feats = {
            "mean_x": 0, "var_x": 0, "mean_y": 0, "var_y": 0,
            "mean_speed": 0, "var_speed": 0, "max_speed": 0,
        }

    feats["detection_rate"] = detection_rate
    for zname in ZONES:
        feats[f"zone_frac_{zname}"] = zone_counts[zname] / n if n else 0.0

    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="folder of raw session CSVs")
    parser.add_argument("--window", type=float, default=5.0, help="window size in seconds")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    raw = load_all_sessions(args.input)
    raw = raw.sort_values("timestamp")

    rows = []
    for (session_file, label), group in raw.groupby(["session_file", "label"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        t0 = group["timestamp"].iloc[0]
        t_end = group["timestamp"].iloc[-1]

        window_start = t0
        while window_start < t_end:
            window_end = window_start + args.window
            window_df = group[(group["timestamp"] >= window_start) & (group["timestamp"] < window_end)]
            if len(window_df) > 0:
                feats = extract_window_features(window_df)
                feats["label"] = label
                feats["session_file"] = session_file
                rows.append(feats)
            window_start = window_end

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} windows to {args.out}")
    print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()
