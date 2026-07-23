"""
Live Prediction Script
------------------------
Loads your trained model and continuously classifies activity in real
time from the live LD2450 stream, using a sliding window of recent
frames -- same feature extraction logic as extract_features.py.

Usage:
    python predict_live.py --port /dev/serial0 --model model.pkl --window_rows 100

Press Ctrl+C to stop.
"""

import argparse
import time
from collections import deque

import joblib
import numpy as np
import serial

FRAME_HEADER = bytes([0xAA, 0xFF, 0x03, 0x00])
FRAME_FOOTER = bytes([0x55, 0xCC])
FRAME_LEN = 30


def decode_signed(lo, hi):
    raw = (hi << 8) | lo
    if hi & 0x80:
        return raw & 0x7FFF
    else:
        return -raw


def parse_frame(frame_bytes):
    targets = []
    for i in range(3):
        chunk = frame_bytes[i * 8:(i + 1) * 8]
        x = decode_signed(chunk[0], chunk[1])
        y = decode_signed(chunk[2], chunk[3])
        speed = decode_signed(chunk[4], chunk[5])
        targets.append((x, y, speed))
    return targets


def primary_target(targets):
    for (x, y, speed) in targets:
        if x != 0 or y != 0:
            return x, y, speed
    return 0, 0, 0


def extract_features_from_window(window):
    """window is a list of (x, y, speed) tuples, zeros included for
    missed detections -- mirrors extract_window_features() in
    extract_features.py so the model sees the same feature shape."""
    xs, ys, speeds, present_flags = [], [], [], []

    for (x, y, speed) in window:
        present = 1 if (x != 0 or y != 0) else 0
        present_flags.append(present)
        if present:
            xs.append(x)
            ys.append(y)
            speeds.append(speed)

    n = len(window)
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
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=256000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--window_rows", type=int, default=100)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    feature_columns = bundle["feature_columns"]

    ser = serial.Serial(args.port, args.baud, timeout=0)
    buf = bytearray()
    window = deque(maxlen=args.window_rows)

    print("Live prediction started. Press Ctrl+C to stop.\n")

    try:
        while True:
            waiting = ser.in_waiting
            if waiting:
                buf.extend(ser.read(waiting))

            new_frame_added = False
            while True:
                hdr_idx = buf.find(FRAME_HEADER)
                if hdr_idx == -1:
                    if len(buf) > 4096:
                        buf.clear()
                    break
                if len(buf) < hdr_idx + FRAME_LEN:
                    break

                frame = buf[hdr_idx:hdr_idx + FRAME_LEN]
                footer = frame[-2:]

                if footer == FRAME_FOOTER:
                    target_data = frame[4:28]
                    targets = parse_frame(target_data)
                    x, y, speed = primary_target(targets)
                    window.append((x, y, speed))
                    new_frame_added = True

                del buf[:hdr_idx + FRAME_LEN]

            if new_frame_added and len(window) == args.window_rows:
                feats = extract_features_from_window(list(window))
                X = np.array([[feats[col] for col in feature_columns]])
                pred = clf.predict(X)[0]
                probs = clf.predict_proba(X)[0]
                confidence = max(probs)
                print(f"Prediction: {pred:12s}  (confidence: {confidence:.2f})          ", end="\r")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
