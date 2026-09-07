#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import time


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


parser = argparse.ArgumentParser()
parser.add_argument("--hand-live-state", required=True)
parser.add_argument("--arm-live-state", required=True)
parser.add_argument("--left-x", type=float, default=0.02)
parser.add_argument("--delay", type=float, default=0.0)
parser.add_argument("--gesture-history", action="store_true")
parser.add_argument("--duration", type=float, default=60.0)
args = parser.parse_args()

hand_path = Path(args.hand_live_state)
arm_path = Path(args.arm_live_state)
started = time.monotonic()
deadline = started + args.duration
sequence = 0
next_tick = time.monotonic()
hand_current = [0.0] * 12
gesture_phases = (
    (30.0, [0.0] * 6),
    (5.0, [0.7] * 6),
    (5.0, [0.15, 0.9, 0.0, 0.0, 0.0, 0.0]),
    (5.0, [0.765, 0.765, 0.675, 0.0, 0.0, 0.0]),
    (30.0, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    (5.0, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    (5.0, [0.15, 0.0, 0.0, 0.0, 0.0, 0.0]),
)
while time.monotonic() < deadline:
    now = time.monotonic()
    if now < next_tick:
        time.sleep(next_tick - now)
    sequence += 1
    timestamp_ns = time.monotonic_ns()
    hand_goal = [0.0] * 6
    if args.gesture_history:
        phase_time = time.monotonic() - started
        for phase_duration, phase_goal in gesture_phases:
            if phase_time < phase_duration:
                hand_goal = phase_goal
                break
            phase_time -= phase_duration
    hand_goal = hand_goal + hand_goal
    hand_current = [
        value + 0.25 * (goal - value)
        for value, goal in zip(hand_current, hand_goal)
    ]
    common = {
        "armed": True,
        "sequence": sequence,
        "monotonic_timestamp": timestamp_ns / 1_000_000_000.0,
        "published_monotonic_ns": timestamp_ns,
        "target_hand_order": ["left", "right"],
    }
    atomic_write(hand_path, {
        **common,
        "schema": "linker_o6_target_v1",
        "reason": "isolated_ab_test",
        "mapping": "visionpro_giovanni_20260831_v1",
        "visionpro_input_calibrated": True,
        "thumb_yaw_hardware_direction_validated": False,
        "target_units": "normalized_0_1",
        "hardware_axis_order": ["thumb_pitch", "thumb_yaw", "index", "middle", "ring", "pinky"],
        "target_12": hand_current,
    })
    atomic_write(arm_path, {
        **common,
        "schema": "r1_a7_arm_ik_target_v1",
        "reason": "isolated_ab_test",
        "mapping": "r1_a7_keyboard_cartesian_v1",
        "reference": "ready_wrist_yaw_pose",
        "frame": "r1_root",
        "target_units": "m",
        "position_offset_m": [args.left_x if time.monotonic() - started >= args.delay else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    next_tick += 1.0 / 30.0
