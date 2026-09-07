#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


STATE_PATTERN = re.compile(r"\[O6 LIVE\] state=(\S+)(?: output=(\S+))?")
METRIC_PATTERN = re.compile(
    r"\[O6 LIVE\] sim_hz=(?P<sim>[0-9.]+) publish_hz=(?P<publish>[0-9.]+) "
    r"source_hz=(?P<source>[0-9.]+) age_ms_p50=(?P<p50>[0-9.]+) "
    r"p95=(?P<p95>[0-9.]+) max=(?P<max>[0-9.]+) missed_seq=(?P<missed>\d+) "
    r"stale_steps=(?P<stale>\d+) invalid_steps=(?P<invalid>\d+)"
)
GPU_FAILURE_PATTERN = re.compile(r"CUDA error|GPU crash|ERROR_DEVICE_LOST|unspecified launch failure", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Run and report the O6 synthetic live-input acceptance test.")
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
    return parser.parse_args()


def run(command, check=True):
    return subprocess.run(command, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


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
    result = run(["docker", "inspect", name])
    inspection = json.loads(result.stdout)[0]
    network_mode = inspection["HostConfig"]["NetworkMode"]
    devices = inspection["HostConfig"].get("Devices") or []
    serial_devices = [device for device in devices if "tty" in str(device).lower() or "serial" in str(device).lower()]
    if network_mode != "none":
        raise RuntimeError(f"Container network mode must be none, found {network_mode}")
    if serial_devices:
        raise RuntimeError("Container has serial devices attached")
    return inspection, len(serial_devices)


def verify_cuda(inspection):
    image = inspection["Config"]["Image"]
    result = run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--network",
            "none",
            "--entrypoint",
            "/isaac-sim/python.sh",
            image,
            "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.ones(1, device='cuda').item())",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("CUDA preflight failed. Reboot Ubuntu before running this validation.\n" + result.stdout)


def prepare_live_directory(name, live_state):
    directory = Path(live_state).parent
    if directory.exists() and directory.stat().st_uid != os.getuid():
        expected_parent = Path(f"/run/user/{os.getuid()}")
        if directory.parent != expected_parent or any(directory.iterdir()):
            raise RuntimeError(f"Cannot safely replace non-user-owned live directory: {directory}")
        inspection = json.loads(run(["docker", "inspect", name]).stdout)[0]
        if inspection["State"]["Running"]:
            run(["docker", "stop", "-t", "10", name])
        directory.rmdir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)


def restart_and_wait_for_consumer(name):
    inspection = json.loads(run(["docker", "inspect", name]).stdout)[0]
    started = datetime.now(timezone.utc)
    if inspection["State"]["Running"]:
        run(["docker", "restart", "-t", "10", name])
    else:
        run(["docker", "start", name])

    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        current = json.loads(run(["docker", "inspect", name]).stdout)[0]
        logs = run(["docker", "logs", "--since", started.isoformat(), name], check=False).stdout
        if GPU_FAILURE_PATTERN.search(logs) or "RuntimeError: CUDA" in logs:
            raise RuntimeError("Isaac consumer failed during CUDA startup.\n" + logs[-4000:])
        if "[O6 LIVE] source=" in logs and "[O6 LIVE] state=" in logs:
            return
        if not current["State"]["Running"]:
            raise RuntimeError("Isaac consumer exited before becoming ready.\n" + logs[-4000:])
        time.sleep(1.0)
    raise RuntimeError("Isaac consumer did not become ready within 90 seconds")


def find_conflicting_writer(live_state):
    result = run(["pgrep", "-af", "teleop_hand_and_arm.py|o6_synthetic_live_source.py"], check=False)
    conflicts = []
    for line in result.stdout.splitlines():
        if str(os.getpid()) in line.split(" ", 1)[0]:
            continue
        if live_state in line and "run_o6_synthetic_validation.py" not in line:
            conflicts.append(line)
    return conflicts


def parse_logs(logs):
    states = []
    metrics = []
    for line in logs.splitlines():
        timestamp = docker_timestamp(line)
        state_match = STATE_PATTERN.search(line)
        if state_match:
            states.append({"timestamp": timestamp, "state": state_match.group(1), "output": state_match.group(2), "line": line})
        metric_match = METRIC_PATTERN.search(line)
        if metric_match:
            values = {key: float(value) for key, value in metric_match.groupdict().items()}
            for key in ("missed", "stale", "invalid"):
                values[key] = int(values[key])
            values["timestamp"] = timestamp
            values["line"] = line
            metrics.append(values)
    return states, metrics


def state_seen(states, expected, started_ns, ended_ns):
    lower = started_ns / 1_000_000_000.0 - 0.1
    upper = ended_ns / 1_000_000_000.0 + 0.25
    return any(item["state"] == expected and item["timestamp"] is not None and lower <= item["timestamp"] <= upper for item in states)


def markdown_report(report, checks):
    status = "PASS" if all(checks.values()) else "FAIL"
    metric = report.get("clean_metric")
    lines = [
        "# O6 synthetic live validation",
        "",
        f"- Result: **{status}**",
        f"- Generated: {report['generated_at']}",
        f"- Isaac container: `{report['container']}`",
        f"- Network mode: `{report['network_mode']}`",
        f"- Serial devices attached: `{report['serial_device_count']}`",
        f"- Recorded frames loaded: `{report['source']['loaded_frames']}`",
        f"- Synthetic source rate: `{report['source']['achieved_hz']:.2f} Hz`",
        "",
        "## Acceptance checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(["", "## Clean live-loop metrics", ""])
    if metric:
        lines.extend(
            [
                "| sim_hz | publish_hz | source_hz | source age p50 | p95 | max | missed_seq | stale_steps | invalid_steps |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| {metric['sim']:.1f} | {metric['publish']:.1f} | {metric['source']:.1f} | "
                    f"{metric['p50']:.1f} ms | {metric['p95']:.1f} ms | {metric['max']:.1f} ms | "
                    f"{metric['missed']} | {metric['stale']} | {metric['invalid']} |"
                ),
            ]
        )
    else:
        lines.append("No clean five-second metric window was captured.")
    lines.extend(
        [
            "",
            "## Deadman",
            "",
            f"- Observed stale/open transition: `{report.get('deadman_ms', 'not observed')} ms` after the last valid write.",
            "- Timeout configured in the Isaac consumer: `250 ms`.",
            "",
            "This validation used only the mounted JSON snapshot and an Isaac container with `network=none`; no DDS or serial path was used.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    inspection, serial_device_count = inspect_container(args.container)
    conflicts = find_conflicting_writer(args.live_state)
    if conflicts:
        raise RuntimeError("Another process is writing the live-state path:\n" + "\n".join(conflicts))
    verify_cuda(inspection)
    prepare_live_directory(args.container, args.live_state)
    restart_and_wait_for_consumer(args.container)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    source_script = Path(__file__).with_name("o6_synthetic_live_source.py")
    with tempfile.NamedTemporaryFile(prefix="o6-synthetic-", suffix=".json", delete=False) as temporary:
        result_path = Path(temporary.name)

    command = [
        sys.executable,
        str(source_script),
        "--recording",
        args.recording,
        "--live-state",
        args.live_state,
        "--mapping",
        args.mapping,
        "--result-json",
        str(result_path),
    ]
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    source_output = []
    for line in process.stdout:
        print(line, end="", flush=True)
        source_output.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Synthetic source failed with exit code {return_code}")

    source_result = json.loads(result_path.read_text(encoding="utf-8"))
    result_path.unlink(missing_ok=True)
    logs = run(
        ["docker", "logs", "--timestamps", "--since", started.isoformat(), args.container],
        check=False,
    ).stdout
    states, metrics = parse_logs(logs)

    baseline_start = source_result["baseline"]["started_wall_ns"] / 1_000_000_000.0
    first_fault_start = source_result["fault_events"][0]["started_wall_ns"] / 1_000_000_000.0
    clean_metrics = [
        metric
        for metric in metrics
        if metric["timestamp"] is not None
        and baseline_start + 5.0 <= metric["timestamp"] <= first_fault_start
        and metric["stale"] == 0
        and metric["invalid"] == 0
    ]
    clean_metric = clean_metrics[-1] if clean_metrics else None

    deadman_state = next(
        (
            item
            for item in states
            if item["state"] == "stale"
            and item["output"] == "open"
            and item["timestamp"] is not None
            and item["timestamp"] >= source_result["last_valid_write_wall_ns"] / 1_000_000_000.0
        ),
        None,
    )
    deadman_ms = None
    if deadman_state:
        deadman_ms = (
            deadman_state["timestamp"] - source_result["last_valid_write_wall_ns"] / 1_000_000_000.0
        ) * 1000.0

    kernel_log = run(["journalctl", "-k", "-b", "--since", started.isoformat(), "--no-pager"], check=False).stdout
    event_by_fault = {event["fault"]: event for event in source_result["fault_events"]}
    fault_checks = {
        "schema rejected": state_seen(states, "schema_rejected", event_by_fault["schema"]["started_wall_ns"], event_by_fault["schema"]["ended_wall_ns"]),
        "mapping rejected": state_seen(states, "mapping_rejected", event_by_fault["mapping"]["started_wall_ns"], event_by_fault["mapping"]["ended_wall_ns"]),
        "NaN rejected": state_seen(states, "invalid_target", event_by_fault["nan"]["started_wall_ns"], event_by_fault["nan"]["ended_wall_ns"]),
        "out-of-range rejected": state_seen(states, "target_out_of_range", event_by_fault["out_of_range"]["started_wall_ns"], event_by_fault["out_of_range"]["ended_wall_ns"]),
        "wrong length rejected": state_seen(states, "invalid_target", event_by_fault["wrong_length"]["started_wall_ns"], event_by_fault["wrong_length"]["ended_wall_ns"]),
    }

    checks = {
        "377 calibrated frames loaded": source_result["loaded_frames"] == 377,
        "synthetic source 27-33 Hz": 27.0 <= source_result["achieved_hz"] <= 33.0,
        "clean metric window captured": clean_metric is not None,
        "sim loop >= 30 Hz": clean_metric is not None and clean_metric["sim"] >= 30.0,
        "publish rate 25-35 Hz": clean_metric is not None and 25.0 <= clean_metric["publish"] <= 35.0,
        "source rate 25-35 Hz": clean_metric is not None and 25.0 <= clean_metric["source"] <= 35.0,
        "latency p95 <= 100 ms": clean_metric is not None and clean_metric["p95"] <= 100.0,
        "latency max < 250 ms": clean_metric is not None and clean_metric["max"] < 250.0,
        "no clean-window missed sequence": clean_metric is not None and clean_metric["missed"] == 0,
        "no clean-window stale/invalid step": clean_metric is not None and clean_metric["stale"] == 0 and clean_metric["invalid"] == 0,
        **fault_checks,
        "deadman changed output to open": deadman_state is not None,
        "deadman transition <= 300 ms": deadman_ms is not None and 0.0 <= deadman_ms <= 300.0,
        "no CUDA/Vulkan device loss during test": GPU_FAILURE_PATTERN.search(logs) is None,
        "no kernel Xid during test": "NVRM: Xid" not in kernel_log,
    }

    report = {
        "schema": "linker_o6_synthetic_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "container": args.container,
        "network_mode": inspection["HostConfig"]["NetworkMode"],
        "serial_device_count": serial_device_count,
        "source": source_result,
        "clean_metric": clean_metric,
        "deadman_ms": deadman_ms,
        "states": states,
        "checks": checks,
        "passed": all(checks.values()),
        "source_output": "".join(source_output),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"o6_synthetic_validation_{stamp}.json"
    markdown_path = report_dir / f"o6_synthetic_validation_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_report(report, checks), encoding="utf-8")
    print(f"[O6 VALIDATION] result={'PASS' if report['passed'] else 'FAIL'}")
    print(f"[O6 VALIDATION] report={markdown_path}")
    print(f"[O6 VALIDATION] json={json_path}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[O6 VALIDATION] BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
