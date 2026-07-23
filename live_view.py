"""
Live Position Viewer
---------------------
Shows the LD2450's tracked target position in real time on a top-down
plot, so you can walk to your bed/desk and see exactly what x/y
coordinates correspond to those locations -- use this to set accurate
ZONES values in extract_features.py.

Usage:
    python live_view.py --port /dev/ttyS0

Close the plot window (or Ctrl+C in the terminal) to stop.

The sensor is at the origin (0,0), at the bottom of the plot, facing
"up" (increasing y = farther away from the sensor). x is left/right.
Walk to your bed, note the coordinates printed in the terminal and
shown on the plot; walk to your desk, do the same. Use those numbers
(with a bit of padding) as your ZONES boundaries.
"""

import argparse
import time
from collections import deque

import matplotlib.pyplot as plt
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=256000)
    parser.add_argument("--trail", type=int, default=100, help="number of recent points to show as a fading trail")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    buf = bytearray()

    trail_x = deque(maxlen=args.trail)
    trail_y = deque(maxlen=args.trail)

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter_trail = ax.scatter([], [], c="lightblue", s=15, alpha=0.4)
    scatter_current = ax.scatter([], [], c="red", s=100, zorder=5)
    text_label = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=11)

    ax.set_xlim(-3000, 3000)
    ax.set_ylim(0, 6000)
    ax.set_xlabel("x (mm) -- left / right")
    ax.set_ylabel("y (mm) -- distance from sensor")
    ax.set_title("LD2450 Live Position (sensor at origin, facing up)")
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.plot(0, 0, marker="^", color="black", markersize=15)  # sensor marker
    ax.grid(True, alpha=0.3)

    print("Move around your room. Watch the terminal and plot for x/y values.")
    print("Walk to your bed, note the coords. Walk to your desk, note those too.")
    print("Ctrl+C or close the plot window to stop.\n")

    try:
        while True:
            chunk = ser.read(64)
            if chunk:
                buf.extend(chunk)

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
                    x, y, speed = targets[0]  # primary target

                    if x != 0 or y != 0:
                        trail_x.append(x)
                        trail_y.append(y)
                        scatter_trail.set_offsets(list(zip(trail_x, trail_y)))
                        scatter_current.set_offsets([[x, y]])
                        text_label.set_text(f"x = {x} mm\ny = {y} mm\nspeed = {speed} cm/s")
                        print(f"x={x:6d}  y={y:6d}  speed={speed:5d}", end="\r")
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()

                del buf[:hdr_idx + FRAME_LEN]

            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
