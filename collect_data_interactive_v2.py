"""
Interactive LD2450 Data Collection Script
-------------------------------------------
Records continuously in one session. Press a key at any time to set
what activity is currently happening -- every frame logged after that
gets tagged with the current label, until you press a different key.

Controls (press while the script is running, no need to hit Enter):
    a  ->  label = "sleeping"    (press when you get into bed)
    b  ->  label = "studying"    (press when you sit down to study)
    c  ->  label = "exercising"  (press when you start moving/exercising)
    q  ->  stop recording and save

Usage:
    python collect_data_interactive.py --port /dev/ttyS0 --out data/session_01.csv

Nothing gets logged until you press a/b/c for the first time -- so you
can start the script, get into position, then press the key once
you're actually doing the activity.

Mac/Linux/Raspberry Pi only (uses termios for instant keypress
detection, not available on Windows).
"""

import argparse
import csv
import os
import select
import sys
import termios
import time
import tty

import serial

FRAME_HEADER = bytes([0xAA, 0xFF, 0x03, 0x00])
FRAME_FOOTER = bytes([0x55, 0xCC])
FRAME_LEN = 30

KEY_LABELS = {
    "a": "sleeping",
    "b": "studying",
    "c": "exercising",
}


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
        resolution = chunk[6] | (chunk[7] << 8)
        targets.append((x, y, speed, resolution))
    return targets


def check_keypress():
    """Non-blocking check for a single keypress. Returns the char or None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=256000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ser = serial.Serial(args.port, args.baud, timeout=0)
    buf = bytearray()

    current_label = None
    rows_written = 0
    label_counts = {v: 0 for v in KEY_LABELS.values()}

    print("Interactive data collection started.")
    print("  a = sleeping   b = studying   c = exercising   q = quit & save")
    print("Nothing is logged until you press a label key for the first time.\n")

    # Put terminal into cbreak mode so single keypresses register instantly,
    # no need to press Enter.
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label",
                "t1_x", "t1_y", "t1_speed", "t1_res",
                "t2_x", "t2_y", "t2_speed", "t2_res",
                "t3_x", "t3_y", "t3_speed", "t3_res",
            ])

            running = True
            while running:
                # Check for a keypress without blocking
                key = check_keypress()
                if key:
                    key = key.lower()
                    if key == "q":
                        running = False
                        break
                    elif key in KEY_LABELS:
                        new_label = KEY_LABELS[key]
                        if new_label != current_label:
                            current_label = new_label
                            print(f"\n>>> Label switched to: {current_label}          ")

                # Read serial data
                waiting = ser.in_waiting
                if waiting:
                    buf.extend(ser.read(waiting))

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

                    if footer == FRAME_FOOTER and current_label is not None:
                        target_data = frame[4:28]
                        targets = parse_frame(target_data)
                        row = [current_label]
                        for (x, y, speed, res) in targets:
                            row.extend([x, y, speed, res])
                        writer.writerow(row)
                        rows_written += 1
                        label_counts[current_label] += 1
                        print(
                            f"[{current_label}] frames logged: {rows_written}   "
                            f"(sleeping={label_counts['sleeping']} "
                            f"studying={label_counts['studying']} "
                            f"exercising={label_counts['exercising']})",
                            end="\r",
                        )

                    del buf[:hdr_idx + FRAME_LEN]

                time.sleep(0.01)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    print(f"\n\nDone. Wrote {rows_written} total frames to {args.out}")
    for label, count in label_counts.items():
        print(f"  {label}: {count} frames")


if __name__ == "__main__":
    main()
