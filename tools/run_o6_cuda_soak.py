#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import re
import statistics
import subprocess
import sys
import time
import traceback

from o6_synthetic_live_source import acquire_writer_lock, load_frames, publish_segment


STATE_PATTERN = re.compile(r"\[O6 LIVE\] state=(\S+)(?: output=(\S+))?")
METRIC_PATTERN = re.compile(
    r"\[O6 LIVE\] sim_hz=(?P<sim>[0-9.]+) publish_hz=(?P<publish>[0-9.]+) "
    r"source_hz=(?P<source>[0-9.]+) age_ms_p50=(?P<p50>[0-9.]+|n/a) "
    r"p95=(?P<p95>[0-9.]+|n/a) max=(?P<max>[0-9.]+|n/a) missed_seq=(?P<missed>\d+) "
    r"stale_steps=(?P<stale>\d+) invalid_steps=(?P<invalid>\d+)"
)
GPU_FAILURE_PATTERN = re.compile(
    r"CUDA error|GPU crash|ERROR_DEVICE_LOST|unspecified launch failure|failed to create primary CUDA context",
    re.IGNORECASE,
)
XID_PATTERN = re.compile(r"NVRM:\s*Xid|Xid\s*\(", re.IGNORECASE)
WRITER_NAMES = {
    "teleop_hand_and_arm.py",
    "o6_synthetic_live_source.py",
    "run_o6_synthetic_validation.py",
    "run_o6_cuda_soak.py",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simulation-only O6 CUDA soak against an existing preview.")
    parser.add_argument(
        "--recording",
        default=(
            "/home/hnh/unitree_r1_dev/xr_teleoperate/teleop/utils/data/"
            "linker_o6_visionpro_calibrated/linker_o6_dry_run_20260831_190505_1788174305469254046.jsonl"
        ),
    )
    parser.add_argument("--live-state", default="/run/user/1000/unitree-o6-live/target.json")
    parser.add_argument("--mapping", default="visionpro_giovanni_20260831_v1")
    parser.add_argument("--container", default="unitree-sim-o6-live-preview")
    parser.add_argument("--report-dir", default="/home/hnh/unitree_r1_dev/reports")
    parser.add_argument("--duration-seconds", type=float, default=1800.0)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--expected-frames", type=int, default=377)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--deadman-limit-ms", type=float, default=300.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def run(command, check=True):
    return subprocess.run(command, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def utc_now():
    return datetime.now(timezone.utc)


def docker_timestamp(line):
    token = line.split(" ", 1)[0]
    if token.endswith("Z") and "." in token:
        whole, fraction = token[:-1].split(".", 1)
        token = f"{whole}.{fraction[:6]}+00:00"
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def inspect_container(name):
    result = run(["docker", "inspect", name], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Existing container {name!r} was not found; this runner never creates containers")
    inspection = json.loads(result.stdout)[0]
    network_mode = inspection["HostConfig"]["NetworkMode"]
    devices = inspection["HostConfig"].get("Devices") or []
    mounts = inspection.get("Mounts") or []
    serial_devices = [item for item in devices if re.search(r"tty|serial", str(item), re.IGNORECASE)]
    serial_mounts = [
        item
        for item in mounts
        if re.search(r"(?:^|/)(?:tty|serial)", f"{item.get('Source', '')} {item.get('Destination', '')}", re.IGNORECASE)
    ]
    if network_mode != "none":
        raise RuntimeError(f"Container network mode must be none, found {network_mode}")
    if serial_devices or serial_mounts:
        raise RuntimeError("Container has a tty/serial device or mount attached")
    if not inspection["State"]["Running"]:
        raise RuntimeError("Existing preview container is not running; this runner never starts it")
    return inspection, serial_devices, serial_mounts


def verify_live_mount(inspection, live_state):
    parent = live_state.parent.resolve()
    if not parent.is_dir():
        raise RuntimeError(f"Live-state directory must already exist: {parent}")
    if parent.stat().st_uid != os.getuid() or not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"Live-state directory must be owned and writable by uid {os.getuid()}: {parent}")
    for mount in inspection.get("Mounts") or []:
        if Path(mount.get("Source", "")).resolve() == parent:
            return mount
    raise RuntimeError(f"Live-state directory is not bind-mounted into the preview container: {parent}")


def find_conflicting_writers():
    conflicts = []
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        if pid == os.getpid():
            continue
        try:
            arguments = [
                value.decode(errors="replace")
                for value in (process_dir / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(argument).name in WRITER_NAMES for argument in arguments):
            conflicts.append(f"{pid} {' '.join(arguments)}")
    return conflicts


def container_logs(name, since):
    return run(["docker", "logs", "--timestamps", "--since", since.isoformat(), name], check=False)


def parse_logs(logs):
    states = []
    metrics = []
    for line in logs.splitlines():
        timestamp = docker_timestamp(line)
        state_match = STATE_PATTERN.search(line)
        if state_match:
            states.append(
                {
                    "timestamp": timestamp,
                    "state": state_match.group(1),
                    "output": state_match.group(2),
                    "line": line,
                }
            )
        metric_match = METRIC_PATTERN.search(line)
        if metric_match:
            values = {}
            for key, value in metric_match.groupdict().items():
                if key in ("missed", "stale", "invalid"):
                    values[key] = int(value)
                else:
                    values[key] = None if value == "n/a" else float(value)
            values["timestamp"] = timestamp
            values["line"] = line
            metrics.append(values)
    return states, metrics


def wait_for_ready(name, inspection, timeout, sample_interval):
    started_at_timestamp = docker_timestamp(inspection["State"]["StartedAt"])
    if started_at_timestamp is None:
        raise RuntimeError("Preview container has an invalid StartedAt timestamp")
    started_at = datetime.fromtimestamp(started_at_timestamp, timezone.utc)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current, _, _ = inspect_container(name)
        if current["Id"] != inspection["Id"] or current["State"]["StartedAt"] != inspection["State"]["StartedAt"]:
            raise RuntimeError("Preview container changed while waiting for readiness")
        result = container_logs(name, started_at)
        if result.returncode != 0:
            raise RuntimeError("Unable to read preview container logs")
        if GPU_FAILURE_PATTERN.search(result.stdout):
            raise RuntimeError("Current preview container has a CUDA/Vulkan fatal error")
        _, metrics = parse_logs(result.stdout)
        latest = metrics[-1] if metrics else None
        latest_age = None if latest is None or latest["timestamp"] is None else time.time() - latest["timestamp"]
        source_ready = "[O6 LIVE] source=" in result.stdout and "device=cuda" in result.stdout
        if source_ready and latest_age is not None and 0.0 <= latest_age <= max(10.0, sample_interval * 2.0):
            return latest
        time.sleep(1.0)
    raise RuntimeError("Preview did not produce a fresh CUDA live-loop metric before the readiness timeout")


def sample_gpu(index):
    result = run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    if result.returncode != 0:
        return {"ok": False, "error": result.stdout.strip()}
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 4:
        return {"ok": False, "error": result.stdout.strip()}
    try:
        return {
            "ok": True,
            "temperature_c": float(fields[0]),
            "utilization_percent": float(fields[1]),
            "memory_used_mib": float(fields[2]),
            "memory_total_mib": float(fields[3]),
        }
    except ValueError:
        return {"ok": False, "error": result.stdout.strip()}


def sample_container(name, expected_id, expected_started_at):
    result = run(["docker", "inspect", name], check=False)
    if result.returncode != 0:
        return {"running": False, "error": result.stdout.strip()}
    inspection = json.loads(result.stdout)[0]
    state = inspection["State"]
    same_instance = inspection["Id"] == expected_id and state["StartedAt"] == expected_started_at
    return {
        "running": bool(state["Running"] and same_instance),
        "same_instance": same_instance,
        "status": state["Status"],
        "oom_killed": state["OOMKilled"],
        "pid": state["Pid"],
    }


def publish_worker(result_queue, live_state, frames, mapping, frequency, duration):
    try:
        write_times = []
        result = publish_segment(
            Path(live_state),
            frames,
            0,
            1,
            mapping,
            frequency,
            duration,
            None,
            write_times,
        )
        achieved_hz = 0.0
        intervals = []
        if len(write_times) > 1:
            achieved_hz = (len(write_times) - 1) / (write_times[-1] - write_times[0])
            intervals = [later - earlier for earlier, later in zip(write_times, write_times[1:])]
        result_queue.put(
            {
                "ok": True,
                "achieved_hz": achieved_hz,
                "interval_s": {
                    "p50": statistics.median(intervals) if intervals else None,
                    "p95": (
                        statistics.quantiles(intervals, n=100, method="inclusive")[94]
                        if len(intervals) > 1
                        else (intervals[0] if intervals else None)
                    ),
                    "max": max(intervals) if intervals else None,
                },
                "result": result,
            }
        )
    except BaseException:
        result_queue.put({"ok": False, "error": traceback.format_exc()})


def number_summary(values):
    if not values:
        return None
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def metric_summary(metrics):
    numeric_age = [item for item in metrics if all(item[key] is not None for key in ("p50", "p95", "max"))]
    timestamps = [item["timestamp"] for item in metrics if item["timestamp"] is not None]
    heartbeat_gaps = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    return {
        "window_count": len(metrics),
        "heartbeat_gap_s": number_summary(heartbeat_gaps),
        "sim_hz": number_summary([item["sim"] for item in metrics]),
        "publish_hz": number_summary([item["publish"] for item in metrics]),
        "source_hz": number_summary([item["source"] for item in metrics]),
        "age_ms_worst": {
            "p50": max((item["p50"] for item in numeric_age), default=None),
            "p95": max((item["p95"] for item in numeric_age), default=None),
            "max": max((item["max"] for item in numeric_age), default=None),
        },
        "missed_seq": {
            "total": sum(item["missed"] for item in metrics),
            "worst_window": max((item["missed"] for item in metrics), default=None),
        },
        "stale_steps": {
            "total": sum(item["stale"] for item in metrics),
            "worst_window": max((item["stale"] for item in metrics), default=None),
        },
        "invalid_steps": {
            "total": sum(item["invalid"] for item in metrics),
            "worst_window": max((item["invalid"] for item in metrics), default=None),
        },
    }


def find_deadman(states, last_write_wall_ns):
    lower = last_write_wall_ns / 1_000_000_000.0
    for item in states:
        if item["timestamp"] is None or item["timestamp"] < lower:
            continue
        if item["state"] == "stale" and item["output"] == "open":
            return (item["timestamp"] - lower) * 1000.0, item
    return None, None


def observe_deadman(container, since, last_write_wall_ns, limit_ms):
    deadline = time.monotonic() + max(1.0, limit_ms / 1000.0 + 0.5)
    latest_logs = ""
    while time.monotonic() < deadline:
        result = container_logs(container, since)
        latest_logs = result.stdout
        states, _ = parse_logs(latest_logs)
        deadman_ms, event = find_deadman(states, last_write_wall_ns)
        if event is not None:
            return deadman_ms, event, latest_logs
        time.sleep(0.02)
    return None, None, latest_logs


def kernel_xids(since):
    result = run(
        ["journalctl", "-k", "-b", "--since", f"@{since.timestamp():.6f}", "--no-pager"],
        check=False,
    )
    if result.returncode != 0:
        return None, result.stdout.strip()
    lines = [line for line in result.stdout.splitlines() if XID_PATTERN.search(line)]
    return lines, None


def format_value(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown(path, report):
    summary = report["metrics_summary"]
    gpu = report["gpu_summary"]
    checks = report["checks"]
    lines = [
        "# O6 CUDA soak",
        "",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Generated: `{report['generated_at']}`",
        f"- Requested duration: `{report['requested_duration_s']:.1f} s`",
        f"- Valid writes: `{report['publisher']['writes']}` at `{report['publisher']['achieved_hz']:.2f} Hz`",
        f"- Container: `{report['container']['name']}` (`network=none`, no tty/serial devices)",
        "",
        "## Acceptance checks",
        "",
        "| Check | Result |",
        "|---|:---:|",
    ]
    for name, passed in checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Consumer metrics",
            "",
            "| Metric | Minimum | Median | Maximum/worst |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ("sim_hz", "publish_hz", "source_hz"):
        item = summary[key] or {}
        lines.append(
            f"| {key} | {format_value(item.get('min'))} | {format_value(item.get('median'))} | {format_value(item.get('max'))} |"
        )
    age = summary["age_ms_worst"]
    lines.append(
        f"| source age ms | p50 {format_value(age['p50'])} | p95 {format_value(age['p95'])} | max {format_value(age['max'])} |"
    )
    lines.extend(
        [
            "",
            f"- Metric windows: `{summary['window_count']}`",
            f"- Missed sequence total/worst: `{summary['missed_seq']['total']}` / `{summary['missed_seq']['worst_window']}`",
            f"- Stale steps total/worst: `{summary['stale_steps']['total']}` / `{summary['stale_steps']['worst_window']}`",
            f"- Invalid steps total/worst: `{summary['invalid_steps']['total']}` / `{summary['invalid_steps']['worst_window']}`",
            f"- Deadman stale/open: `{format_value(report['deadman']['elapsed_ms'])} ms`",
            "",
            "## GPU telemetry",
            "",
            f"- Samples: `{gpu['sample_count']}`",
            f"- Temperature min/median/max: `{format_value((gpu['temperature_c'] or {}).get('min'))}` / "
            f"`{format_value((gpu['temperature_c'] or {}).get('median'))}` / `{format_value((gpu['temperature_c'] or {}).get('max'))} °C`",
            f"- Utilization min/median/max: `{format_value((gpu['utilization_percent'] or {}).get('min'))}` / "
            f"`{format_value((gpu['utilization_percent'] or {}).get('median'))}` / `{format_value((gpu['utilization_percent'] or {}).get('max'))} %`",
            f"- Memory used min/median/max: `{format_value((gpu['memory_used_mib'] or {}).get('min'))}` / "
            f"`{format_value((gpu['memory_used_mib'] or {}).get('median'))}` / `{format_value((gpu['memory_used_mib'] or {}).get('max'))} MiB`",
            "",
            "Only the bind-mounted atomic JSON snapshot was published. The preview has `network=none`; no DDS or serial path is used.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    positive = (args.duration_seconds, args.frequency, args.sample_interval, args.ready_timeout, args.deadman_limit_ms)
    if not all(math.isfinite(value) and value > 0.0 for value in positive):
        raise RuntimeError("Duration, frequency, intervals, and limits must be positive finite values")
    if args.expected_frames <= 0 or args.gpu_index < 0:
        raise RuntimeError("Expected frames must be positive and GPU index must be non-negative")

    live_state = Path(args.live_state).expanduser().resolve()
    inspection, serial_devices, serial_mounts = inspect_container(args.container)
    live_mount = verify_live_mount(inspection, live_state)
    conflicts = find_conflicting_writers()
    if conflicts:
        raise RuntimeError("Another O6 live-state writer is running:\n" + "\n".join(conflicts))
    ready_metric = wait_for_ready(args.container, inspection, args.ready_timeout, args.sample_interval)
    frames = load_frames(args.recording, args.mapping, args.expected_frames)
    writer_lock = acquire_writer_lock(live_state, "run_o6_cuda_soak")

    started = utc_now()
    monitor_started_monotonic = time.monotonic()
    result_queue = multiprocessing.Queue()
    publisher = multiprocessing.Process(
        target=publish_worker,
        args=(result_queue, str(live_state), frames, args.mapping, args.frequency, args.duration_seconds),
        name="o6-soak-valid-publisher",
    )
    publisher.start()
    health_samples = []
    monitor_errors = []
    interrupted = False
    next_sample = monitor_started_monotonic
    try:
        while publisher.is_alive():
            now = time.monotonic()
            if now >= next_sample:
                logs_result = container_logs(args.container, started)
                _, metrics = parse_logs(logs_result.stdout)
                container = sample_container(args.container, inspection["Id"], inspection["State"]["StartedAt"])
                gpu = sample_gpu(args.gpu_index)
                latest_metric = metrics[-1] if metrics else None
                metric_age_s = (
                    None
                    if latest_metric is None or latest_metric["timestamp"] is None
                    else time.time() - latest_metric["timestamp"]
                )
                sample = {
                    "timestamp": utc_now().isoformat(),
                    "elapsed_s": now - monitor_started_monotonic,
                    "container": container,
                    "gpu": gpu,
                    "latest_metric": latest_metric,
                    "latest_metric_age_s": metric_age_s,
                    "logs_ok": logs_result.returncode == 0,
                }
                health_samples.append(sample)
                if not sample["logs_ok"]:
                    monitor_errors.append("docker logs failed")
                if not container.get("running"):
                    monitor_errors.append("preview container stopped or changed")
                if not gpu.get("ok"):
                    monitor_errors.append("nvidia-smi telemetry failed")
                if sample["elapsed_s"] >= args.sample_interval * 2.0 and (
                    metric_age_s is None or metric_age_s > max(15.0, args.sample_interval * 3.0)
                ):
                    monitor_errors.append("consumer metric heartbeat became stale")
                metric_text = "n/a" if latest_metric is None else f"{latest_metric['sim']:.1f}Hz"
                gpu_text = "n/a" if not gpu.get("ok") else f"{gpu['temperature_c']:.0f}C/{gpu['utilization_percent']:.0f}%"
                print(
                    f"[O6 SOAK] elapsed={sample['elapsed_s']:.1f}s sim={metric_text} "
                    f"container={'up' if container.get('running') else 'down'} gpu={gpu_text}",
                    flush=True,
                )
                if monitor_errors:
                    break
                while next_sample <= now:
                    next_sample += args.sample_interval
            time.sleep(min(0.2, max(0.01, next_sample - time.monotonic())))
    except KeyboardInterrupt:
        interrupted = True
        monitor_errors.append("interrupted")
    finally:
        if publisher.is_alive():
            publisher.terminate()
        publisher.join(timeout=5.0)

    try:
        publisher_result = result_queue.get(timeout=1.0)
    except queue.Empty:
        publisher_result = {"ok": False, "error": "publisher did not complete"}
    if not publisher_result.get("ok"):
        monitor_errors.append("publisher failed: " + publisher_result.get("error", "unknown error"))

    final_logs_result = container_logs(args.container, started)
    final_logs = final_logs_result.stdout
    if final_logs_result.returncode != 0:
        monitor_errors.append("final docker logs failed")
    deadman_ms = None
    deadman_event = None
    publish_data = publisher_result.get("result") or {}
    last_write_wall_ns = publish_data.get("last_write_wall_ns")
    if last_write_wall_ns is not None:
        deadman_ms, deadman_event, final_logs = observe_deadman(
            args.container, started, last_write_wall_ns, args.deadman_limit_ms
        )

    states, all_metrics = parse_logs(final_logs)
    if publish_data:
        first_clean_timestamp = publish_data["started_wall_ns"] / 1_000_000_000.0 + args.sample_interval
        last_publish_timestamp = publish_data["last_write_wall_ns"] / 1_000_000_000.0
        soak_metrics = [
            metric
            for metric in all_metrics
            if metric["timestamp"] is not None
            and first_clean_timestamp <= metric["timestamp"] <= last_publish_timestamp
            and all(metric[key] is not None for key in ("p50", "p95", "max"))
        ]
    else:
        soak_metrics = []
    summary = metric_summary(soak_metrics)
    final_container = sample_container(args.container, inspection["Id"], inspection["State"]["StartedAt"])
    gpu_samples = [sample["gpu"] for sample in health_samples if sample["gpu"].get("ok")]
    gpu_summary = {
        "sample_count": len(gpu_samples),
        "temperature_c": number_summary([item["temperature_c"] for item in gpu_samples]),
        "utilization_percent": number_summary([item["utilization_percent"] for item in gpu_samples]),
        "memory_used_mib": number_summary([item["memory_used_mib"] for item in gpu_samples]),
    }
    xids, xid_error = kernel_xids(started)
    fatal_lines = [line for line in final_logs.splitlines() if GPU_FAILURE_PATTERN.search(line)]

    sim = summary["sim_hz"] or {}
    publish = summary["publish_hz"] or {}
    source = summary["source_hz"] or {}
    age = summary["age_ms_worst"]
    heartbeat = summary["heartbeat_gap_s"] or {}
    container_stayed_running = bool(health_samples) and all(
        sample["container"].get("running", False) for sample in health_samples
    ) and final_container.get("running", False)
    checks = {
        "publisher completed requested duration": publisher_result.get("ok", False) and not interrupted,
        "publisher achieved 27-33 Hz": publisher_result.get("ok", False)
        and 27.0 <= publisher_result.get("achieved_hz", 0.0) <= 33.0,
        "clean consumer metric windows captured": summary["window_count"] > 0,
        "consumer metric heartbeat remained continuous": summary["window_count"] > 0
        and heartbeat.get("max", 0.0) <= max(15.0, args.sample_interval * 3.0),
        "sim loop minimum >= 30 Hz": sim.get("min", 0.0) >= 30.0,
        "publish rate stayed 25-35 Hz": publish.get("min", 0.0) >= 25.0 and publish.get("max", math.inf) <= 35.0,
        "source rate stayed 25-35 Hz": source.get("min", 0.0) >= 25.0 and source.get("max", math.inf) <= 35.0,
        "source age p95 worst <= 100 ms": age["p95"] is not None and age["p95"] <= 100.0,
        "source age max worst < 250 ms": age["max"] is not None and age["max"] < 250.0,
        "no missed sequence": summary["missed_seq"]["total"] == 0,
        "no stale or invalid soak step": summary["stale_steps"]["total"] == 0
        and summary["invalid_steps"]["total"] == 0,
        "container stayed alive and unchanged": container_stayed_running,
        "monitor completed without error": not monitor_errors,
        "GPU telemetry captured every monitor cycle": bool(health_samples) and len(gpu_samples) == len(health_samples),
        "no CUDA/Vulkan fatal log": not fatal_lines,
        "no kernel Xid": xids == [],
        "deadman stale/open observed <= limit": deadman_ms is not None and 0.0 <= deadman_ms <= args.deadman_limit_ms,
    }

    report = {
        "schema": "linker_o6_cuda_soak_v1",
        "generated_at": utc_now().isoformat(),
        "started_at": started.isoformat(),
        "requested_duration_s": args.duration_seconds,
        "sample_interval_s": args.sample_interval,
        "recording": str(Path(args.recording)),
        "loaded_frames": len(frames),
        "live_state": str(live_state),
        "mapping": args.mapping,
        "container": {
            "name": args.container,
            "id": inspection["Id"],
            "image": inspection["Config"]["Image"],
            "started_at": inspection["State"]["StartedAt"],
            "network_mode": inspection["HostConfig"]["NetworkMode"],
            "serial_devices": serial_devices,
            "serial_mounts": serial_mounts,
            "live_mount": live_mount,
            "ready_metric": ready_metric,
            "final_state": final_container,
        },
        "publisher": {
            "ok": publisher_result.get("ok", False),
            "requested_hz": args.frequency,
            "achieved_hz": publisher_result.get("achieved_hz", 0.0),
            "interval_s": publisher_result.get("interval_s"),
            "writes": publish_data.get("writes", 0),
            "started_wall_ns": publish_data.get("started_wall_ns"),
            "last_write_wall_ns": last_write_wall_ns,
            "error": publisher_result.get("error"),
        },
        "metrics_summary": summary,
        "metric_windows": soak_metrics,
        "health_samples": health_samples,
        "gpu_summary": gpu_summary,
        "deadman": {"limit_ms": args.deadman_limit_ms, "elapsed_ms": deadman_ms, "event": deadman_event},
        "cuda_vulkan_errors": fatal_lines,
        "kernel_xids": xids,
        "kernel_log_error": xid_error,
        "monitor_errors": monitor_errors,
        "states": states,
        "checks": checks,
        "passed": all(checks.values()),
        "safety": "Atomic JSON IPC only; container network=none; no DDS or serial device path.",
    }
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"o6_cuda_soak_{stamp}.json"
    markdown_path = report_dir / f"o6_cuda_soak_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(markdown_path, report)
    print(f"[O6 SOAK] result={'PASS' if report['passed'] else 'FAIL'}")
    print(f"[O6 SOAK] report={markdown_path}")
    print(f"[O6 SOAK] json={json_path}")
    writer_lock.close()
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[O6 SOAK] BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
