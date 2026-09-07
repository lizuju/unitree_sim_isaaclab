#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import time


SCHEMA = "linker_o6_target_v1"
AXIS_ORDER = ["thumb_pitch", "thumb_yaw", "index", "middle", "ring", "pinky"]
HAND_ORDER = ["left", "right"]
FAULTS = (
    ("schema", "schema_rejected"),
    ("mapping", "mapping_rejected"),
    ("nan", "invalid_target"),
    ("out_of_range", "target_out_of_range"),
    ("wrong_length", "invalid_target"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Publish recorded O6 targets as a synthetic 30 Hz live source.")
    parser.add_argument("--recording", required=True)
    parser.add_argument("--live-state", required=True)
    parser.add_argument("--mapping", default="visionpro_giovanni_20260831_v1")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--expected-frames", type=int, default=377)
    parser.add_argument("--baseline-seconds", type=float)
    parser.add_argument("--fault-seconds", type=float, default=0.5)
    parser.add_argument("--recovery-seconds", type=float, default=0.5)
    parser.add_argument("--final-valid-seconds", type=float, default=2.0)
    parser.add_argument("--deadman-wait-seconds", type=float, default=0.7)
    parser.add_argument("--result-json", required=True)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_frames(path, mapping, expected_frames):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    frames_by_timestamp = {}
    for row in rows:
        if not row.get("armed") or "target_12" not in row:
            continue
        if row.get("mapping") != mapping:
            raise ValueError("Recording mapping does not match --mapping")
        if row.get("visionpro_input_calibrated") is not True:
            raise ValueError("Recording is not Vision Pro calibrated")
        if row.get("thumb_yaw_hardware_direction_validated") is not False:
            raise ValueError("Recording must remain hardware-direction unvalidated")
        target = row["target_12"]
        if not isinstance(target, list) or len(target) != 12:
            raise ValueError("Recording target must contain exactly 12 values")
        target = [float(value) for value in target]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in target):
            raise ValueError("Recording targets must be finite normalized values")
        timestamp = float(row["monotonic_timestamp"])
        if not math.isfinite(timestamp):
            raise ValueError("Recording timestamp must be finite")
        frames_by_timestamp[timestamp] = target

    frames = [frames_by_timestamp[timestamp] for timestamp in sorted(frames_by_timestamp)]
    if len(frames) != expected_frames:
        raise ValueError(f"Expected {expected_frames} unique frames, found {len(frames)}")
    return frames


def atomic_write_json(path, payload, allow_nan=False, secure_parent=True):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if secure_parent:
        os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=allow_nan)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def acquire_writer_lock(live_state, writer_name):
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


def make_payload(target, sequence, mapping, fault):
    target = list(target)
    timestamp_ns = time.monotonic_ns()
    payload = {
        "schema": SCHEMA,
        "armed": True,
        "reason": "synthetic_replay",
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
    if fault == "schema":
        payload["schema"] = "linker_o6_target_fault"
    elif fault == "mapping":
        payload["mapping"] = mapping + "_fault"
    elif fault == "nan":
        payload["target_12"][0] = float("nan")
    elif fault == "out_of_range":
        payload["target_12"][0] = 1.25
    elif fault == "wrong_length":
        payload["target_12"] = payload["target_12"][:-1]
    return payload


def publish_segment(path, frames, frame_index, sequence, mapping, frequency, duration, fault, write_times):
    started_monotonic = time.monotonic()
    started_wall_ns = time.time_ns()
    writes = 0
    last_write_wall_ns = None
    frame_count = max(1, round(duration * frequency))
    for tick in range(frame_count):
        deadline = started_monotonic + tick / frequency
        delay = deadline - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        payload = make_payload(frames[frame_index], sequence, mapping, fault)
        atomic_write_json(path, payload, allow_nan=fault == "nan")
        last_write_wall_ns = time.time_ns()
        write_times.append(time.monotonic())
        writes += 1
        sequence += 1
        frame_index = (frame_index + 1) % len(frames)
    remaining = started_monotonic + duration - time.monotonic()
    if remaining > 0.0:
        time.sleep(remaining)
    return {
        "frame_index": frame_index,
        "sequence": sequence,
        "writes": writes,
        "started_wall_ns": started_wall_ns,
        "ended_wall_ns": time.time_ns(),
        "last_write_wall_ns": last_write_wall_ns,
    }


def main():
    args = parse_args()
    positive_values = (
        args.frequency,
        args.fault_seconds,
        args.recovery_seconds,
        args.final_valid_seconds,
        args.deadman_wait_seconds,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in positive_values):
        raise ValueError("Frequency and durations must be positive finite values")
    if args.baseline_seconds is not None and not (
        math.isfinite(args.baseline_seconds) and args.baseline_seconds > 0.0
    ):
        raise ValueError("--baseline-seconds must be a positive finite value")

    live_state = Path(args.live_state)
    result_path = Path(args.result_json)
    frames = load_frames(args.recording, args.mapping, args.expected_frames)
    writer_lock = acquire_writer_lock(live_state, "o6_synthetic_live_source")
    baseline_seconds = args.baseline_seconds if args.baseline_seconds is not None else len(frames) / args.frequency
    print(
        f"[O6 SYNTHETIC] loaded_frames={len(frames)} frequency={args.frequency:.1f}Hz "
        f"live_state={live_state}",
        flush=True,
    )

    frame_index = 0
    sequence = 1
    write_times = []
    events = []

    baseline = publish_segment(
        live_state,
        frames,
        frame_index,
        sequence,
        args.mapping,
        args.frequency,
        baseline_seconds,
        None,
        write_times,
    )
    frame_index = baseline["frame_index"]
    sequence = baseline["sequence"]
    print(f"[O6 SYNTHETIC] baseline_complete writes={baseline['writes']} utc={utc_now()}", flush=True)

    for fault, expected_state in FAULTS:
        event = publish_segment(
            live_state,
            frames,
            frame_index,
            sequence,
            args.mapping,
            args.frequency,
            args.fault_seconds,
            fault,
            write_times,
        )
        frame_index = event["frame_index"]
        sequence = event["sequence"]
        event.update({"fault": fault, "expected_state": expected_state})
        events.append(event)
        print(
            f"[O6 SYNTHETIC] fault={fault} expected_state={expected_state} writes={event['writes']} utc={utc_now()}",
            flush=True,
        )
        recovery = publish_segment(
            live_state,
            frames,
            frame_index,
            sequence,
            args.mapping,
            args.frequency,
            args.recovery_seconds,
            None,
            write_times,
        )
        frame_index = recovery["frame_index"]
        sequence = recovery["sequence"]

    final_valid = publish_segment(
        live_state,
        frames,
        frame_index,
        sequence,
        args.mapping,
        args.frequency,
        args.final_valid_seconds,
        None,
        write_times,
    )
    last_write_wall_ns = final_valid["last_write_wall_ns"]
    stopped_wall_ns = time.time_ns()
    print(
        f"[O6 SYNTHETIC] writes_stopped deadman_wait={args.deadman_wait_seconds:.3f}s utc={utc_now()}",
        flush=True,
    )
    time.sleep(args.deadman_wait_seconds)

    achieved_hz = 0.0
    if len(write_times) > 1:
        achieved_hz = (len(write_times) - 1) / (write_times[-1] - write_times[0])
    result = {
        "schema": "linker_o6_synthetic_result_v1",
        "recording": str(Path(args.recording)),
        "live_state": str(live_state),
        "mapping": args.mapping,
        "loaded_frames": len(frames),
        "requested_hz": args.frequency,
        "achieved_hz": achieved_hz,
        "total_writes": len(write_times),
        "baseline": baseline,
        "fault_events": events,
        "last_valid_write_wall_ns": last_write_wall_ns,
        "writes_stopped_wall_ns": stopped_wall_ns,
        "completed_wall_ns": time.time_ns(),
    }
    atomic_write_json(result_path, result, secure_parent=False)
    writer_lock.close()
    print(f"[O6 SYNTHETIC] complete achieved_hz={achieved_hz:.2f} result={result_path}", flush=True)


if __name__ == "__main__":
    main()
