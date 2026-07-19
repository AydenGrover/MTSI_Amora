"""
LD2450 Data Collection Script
------------------------------
Reads live target data from the LD2450 over serial and logs it to CSV,
tagged with whatever activity label you're currently performing.

Usage:
    python collect_data.py --port /dev/tty.usbserial-XXXX --label sleeping --duration 600 --out data/sleeping_01.csv

    --port      Serial port the LD2450 is connected to
                (Mac: something like /dev/tty.usbserial-XXXX or /dev/cu.usbserial-XXXX
                 Windows: COM3, COM4, etc.
                 Linux: /dev/ttyUSB0)
    --label     Activity label for this session: sleeping / studying / exercising
    --duration  How many seconds to record (e.g. 600 = 10 minutes)
    --out       Output CSV path

Run this once per activity per session. Do several separate sessions per
activity (different times of day, slightly different positioning) rather
than one long session -- it makes your model more robust.
"""

import argparse
import csv
import os
import time

import serial

FRAME_HEADER = bytes([0xAA, 0xFF, 0x03, 0x00])
FRAME_FOOTER = bytes([0x55, 0xCC])
FRAME_LEN = 30  # 4 header + 24 target data + 2 footer


def decode_signed(lo, hi):
    """LD2450 encodes sign in the top bit of the high byte.
    If that bit is SET -> value is positive (magnitude = raw & 0x7FFF).
    If that bit is CLEAR -> value is negative (magnitude = -raw)."""
    raw = (hi << 8) | lo
    if hi & 0x80:
        return raw & 0x7FFF
    else:
        return -raw


def parse_frame(frame_bytes):
    """frame_bytes is the 24-byte target data block (3 targets x 8 bytes)."""
    targets = []
    for i in range(3):
        chunk = frame_bytes[i * 8:(i + 1) * 8]
        x = decode_signed(chunk[0], chunk[1])
        y = decode_signed(chunk[2], chunk[3])
        speed = decode_signed(chunk[4], chunk[5])
        resolution = chunk[6] | (chunk[7] << 8)
        targets.append((x, y, speed, resolution))
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=256000)
    parser.add_argument("--label", required=True)
    parser.add_argument("--duration", type=int, default=600, help="seconds")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ser = serial.Serial(args.port, args.baud, timeout=1)
    buf = bytearray()

    print(f"Recording '{args.label}' for {args.duration}s -> {args.out}")
    print("Press Ctrl+C to stop early.")

    start = time.time()
    rows_written = 0

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "label",
            "t1_x", "t1_y", "t1_speed", "t1_res",
            "t2_x", "t2_y", "t2_speed", "t2_res",
            "t3_x", "t3_y", "t3_speed", "t3_res",
        ])

        try:
            while time.time() - start < args.duration:
                chunk = ser.read(64)
                if chunk:
                    buf.extend(chunk)

                # Look for a complete frame in the buffer
                while True:
                    hdr_idx = buf.find(FRAME_HEADER)
                    if hdr_idx == -1:
                        # keep buffer from growing unbounded
                        if len(buf) > 4096:
                            buf.clear()
                        break
                    if len(buf) < hdr_idx + FRAME_LEN:
                        break  # wait for more data

                    frame = buf[hdr_idx:hdr_idx + FRAME_LEN]
                    footer = frame[-2:]

                    if footer == FRAME_FOOTER:
                        target_data = frame[4:28]
                        targets = parse_frame(target_data)
                        ts = time.time()
                        row = [ts, args.label]
                        for (x, y, speed, res) in targets:
                            row.extend([x, y, speed, res])
                        writer.writerow(row)
                        rows_written += 1
                        if rows_written % 20 == 0:
                            elapsed = int(time.time() - start)
                            print(f"  {elapsed}s elapsed, {rows_written} frames logged", end="\r")

                    # consume up through this frame and keep scanning
                    del buf[:hdr_idx + FRAME_LEN]

        except KeyboardInterrupt:
            print("\nStopped early by user.")

    print(f"\nDone. Wrote {rows_written} frames to {args.out}")


if __name__ == "__main__":
    main()
