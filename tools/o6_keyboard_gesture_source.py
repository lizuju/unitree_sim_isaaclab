#!/usr/bin/env python3
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

from r1_a7_workspace import constrain_dual_offset


SCHEMA = "linker_o6_target_v1"
MAPPING = "visionpro_giovanni_20260831_v1"
AXIS_ORDER = ["thumb_pitch", "thumb_yaw", "index", "middle", "ring", "pinky"]
HAND_ORDER = ["left", "right"]
GESTURES = {
    "open": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "fist": [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    "thumb_abduction": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
    "thumb_adduction": [0.15, 0.9, 0.0, 0.0, 0.0, 0.0],
    "thumb_index_touch": [0.765, 0.765, 0.675, 0.0, 0.0, 0.0],
    "ring": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "pinky": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}
GESTURE_KEYS = {
    "o": "open",
    "f": "fist",
    "a": "thumb_abduction",
    "d": "thumb_adduction",
    "t": "thumb_index_touch",
    "r": "ring",
    "p": "pinky",
}
HAND_KEYS = {"1": "left", "2": "right", "3": "both"}
ARM_HAND_KEYS = {"4": "left", "5": "right", "6": "both"}
ARM_MOVE_KEYS = {
    "i": (0, 0.005),
    "k": (0, -0.005),
    "j": (1, 0.005),
    "l": (1, -0.005),
    "u": (2, 0.005),
    "n": (2, -0.005),
}
ARM_SCHEMA = "r1_a7_arm_ik_target_v1"
ARM_MAPPING = "r1_a7_keyboard_cartesian_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Publish Linker O6 gestures from a terminal keyboard.")
    parser.add_argument("--live-state", default="/run/user/1000/unitree-o6-live/target.json")
    parser.add_argument("--mapping", default=MAPPING)
    parser.add_argument("--arm-live-state")
    parser.add_argument("--arm-mapping", default=ARM_MAPPING)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.25,
        help="Fraction of the remaining target distance applied per 30 Hz tick (0 < value <= 1).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print status without writing the JSON file.")
    return parser.parse_args()


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def acquire_writer_lock(live_state, writer_name="o6_keyboard_gesture_source"):
    live_state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(live_state.parent, 0o700)
    lock_path = live_state.parent / "writer.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    os.chmod(lock_path, 0o600)
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError(f"Another O6 live-state writer holds {lock_path}")
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()} writer={writer_name}\n")
    lock_file.flush()
    return lock_file


def make_payload(target, sequence, mapping, timestamp_ns=None):
    if timestamp_ns is None:
        timestamp_ns = time.monotonic_ns()
    return {
        "schema": SCHEMA,
        "armed": True,
        "reason": "keyboard_gesture",
        "mapping": mapping,
        "visionpro_input_calibrated": True,
        "thumb_yaw_hardware_direction_validated": False,
        "target_units": "normalized_0_1",
        "monotonic_timestamp": timestamp_ns / 1_000_000_000.0,
        "hardware_axis_order": AXIS_ORDER,
        "target_hand_order": HAND_ORDER,
        "target_12": target,
        "sequence": sequence,
        "published_monotonic_ns": timestamp_ns,
    }


def make_arm_payload(offset, sequence, mapping, timestamp_ns=None):
    if timestamp_ns is None:
        timestamp_ns = time.monotonic_ns()
    return {
        "schema": ARM_SCHEMA,
        "armed": True,
        "reason": "keyboard_cartesian",
        "mapping": mapping,
        "reference": "armed_actual_wrist_yaw_pose",
        "frame": "r1_waist_yaw",
        "target_units": "m",
        "target_hand_order": HAND_ORDER,
        "position_offset_m": offset,
        "sequence": sequence,
        "monotonic_timestamp": timestamp_ns / 1_000_000_000.0,
        "published_monotonic_ns": timestamp_ns,
    }


def set_selected_hand(target, selected_hand, gesture):
    if selected_hand in ("left", "both"):
        target[:6] = gesture
    if selected_hand in ("right", "both"):
        target[6:] = gesture


def move_selected_arm(target, selected_arm, axis, delta):
    for base in (0, 3):
        side = "left" if base == 0 else "right"
        if selected_arm in (side, "both"):
            index = base + axis
            target[index] += delta
    target[:], _ = constrain_dual_offset(target)


def center_selected_arm(target, selected_arm):
    for base in (0, 3):
        side = "left" if base == 0 else "right"
        if selected_arm in (side, "both"):
            target[base : base + 3] = [0.0, 0.0, 0.0]


def print_help(arm_enabled):
    print("[O6 KEYBOARD] hand: 1=left 2=right 3=both")
    print("[O6 KEYBOARD] gesture: o=open f=fist a=thumb-abduction d=thumb-adduction")
    print("[O6 KEYBOARD]          t=thumb-index-touch r=ring p=pinky q=quit")
    if arm_enabled:
        print("[R1 IK KEYBOARD] arm: 4=left 5=right 6=both 0=center")
        print("[R1 IK KEYBOARD] move: i/k=forward/back j/l=left/right u/n=up/down (5 mm)")


def read_available_keys():
    keys = []
    stdin_fd = sys.stdin.fileno()
    while select.select([stdin_fd], [], [], 0.0)[0]:
        key = os.read(stdin_fd, 1)
        if not key:
            break
        keys.append(key.decode(errors="ignore").lower())
    return keys


def main():
    args = parse_args()
    if not math.isfinite(args.frequency) or args.frequency <= 0.0:
        raise ValueError("--frequency must be positive and finite")
    if not math.isfinite(args.smoothing) or not 0.0 < args.smoothing <= 1.0:
        raise ValueError("--smoothing must be finite and in (0, 1]")

    live_state = Path(args.live_state)
    selected_hand = "both"
    selected_arm = "both"
    selected_gesture = "open"
    desired = [0.0] * 12
    current = [0.0] * 12
    arm_desired = [0.0] * 6
    arm_current = [0.0] * 6
    sequence = 0
    period = 1.0 / args.frequency
    next_tick = time.monotonic()
    next_status = next_tick
    old_terminal = None
    writer_lock = None
    arm_writer_lock = None
    arm_live_state = Path(args.arm_live_state) if args.arm_live_state else None

    print_help(arm_live_state is not None)
    mode = "DRY RUN (no file writes)" if args.dry_run else f"writing {live_state}"
    print(
        f"[O6 KEYBOARD] {mode}; frequency={args.frequency:.1f}Hz "
        f"smoothing={args.smoothing:.3f}; DDS/network/serial are not used",
        flush=True,
    )

    try:
        if not args.dry_run:
            writer_lock = acquire_writer_lock(live_state)
            if arm_live_state is not None:
                try:
                    arm_writer_lock = acquire_writer_lock(
                        arm_live_state, "o6_keyboard_gesture_source:arm_ik"
                    )
                except Exception:
                    writer_lock.close()
                    writer_lock = None
                    raise
        if sys.stdin.isatty():
            old_terminal = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())

        running = True
        while running:
            for key in read_available_keys():
                if key == "q" or key == "\x03":
                    running = False
                    break
                if key in HAND_KEYS:
                    selected_hand = HAND_KEYS[key]
                    print(f"[O6 KEYBOARD] selected_hand={selected_hand}", flush=True)
                    continue
                if arm_live_state is not None and key in ARM_HAND_KEYS:
                    selected_arm = ARM_HAND_KEYS[key]
                    print(f"[R1 IK KEYBOARD] selected_arm={selected_arm}", flush=True)
                    continue
                if arm_live_state is not None and key in ARM_MOVE_KEYS:
                    axis, delta = ARM_MOVE_KEYS[key]
                    move_selected_arm(arm_desired, selected_arm, axis, delta)
                    print(
                        f"[R1 IK KEYBOARD] arm={selected_arm} desired_offset_m="
                        f"{[round(value, 3) for value in arm_desired]}",
                        flush=True,
                    )
                    continue
                if arm_live_state is not None and key == "0":
                    center_selected_arm(arm_desired, selected_arm)
                    print(
                        f"[R1 IK KEYBOARD] arm={selected_arm} desired_offset_m="
                        f"{[round(value, 3) for value in arm_desired]}",
                        flush=True,
                    )
                    continue
                if key in GESTURE_KEYS:
                    selected_gesture = GESTURE_KEYS[key]
                    set_selected_hand(desired, selected_hand, GESTURES[selected_gesture])
                    print(
                        f"[O6 KEYBOARD] gesture={selected_gesture} hand={selected_hand} "
                        f"desired={[round(value, 3) for value in desired]}",
                        flush=True,
                    )

            if not running:
                break

            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.01))
                continue

            current = [
                value + args.smoothing * (goal - value)
                for value, goal in zip(current, desired)
            ]
            arm_current = [
                value + args.smoothing * (goal - value)
                for value, goal in zip(arm_current, arm_desired)
            ]
            sequence += 1
            timestamp_ns = time.monotonic_ns()
            payload = make_payload(current, sequence, args.mapping, timestamp_ns)
            if not args.dry_run:
                atomic_write_json(live_state, payload)
            if arm_live_state is not None:
                arm_payload = make_arm_payload(
                    arm_current, sequence, args.arm_mapping, timestamp_ns
                )
                if not args.dry_run:
                    atomic_write_json(arm_live_state, arm_payload)

            if now >= next_status:
                action = "would_publish" if args.dry_run else "published"
                print(
                    f"[O6 KEYBOARD] {action} sequence={sequence} gesture={selected_gesture} "
                    f"hand={selected_hand} target={[round(value, 3) for value in current]}",
                    flush=True,
                )
                if arm_live_state is not None:
                    print(
                        f"[R1 IK KEYBOARD] {action} sequence={sequence} arm={selected_arm} "
                        f"offset_m={[round(value, 3) for value in arm_current]}",
                        flush=True,
                    )
                next_status = now + 2.0

            next_tick += period
            if next_tick < now - period:
                next_tick = now + period
    finally:
        if old_terminal is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_terminal)
        if writer_lock is not None:
            writer_lock.close()
        if arm_writer_lock is not None:
            arm_writer_lock.close()

    print("[O6 KEYBOARD] stopped writing; the 250 ms consumer deadman can now take over", flush=True)


if __name__ == "__main__":
    main()
