import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate O6 simulated deadman open settling in Isaac Sim 5.1.")
parser.add_argument("--left-urdf", default="/home/code/linkerhand-urdf/O6/left/linkerhand_o6_left.urdf")
parser.add_argument("--right-urdf", default="/home/code/linkerhand-urdf/O6/right/linkerhand_o6_right.urdf")
parser.add_argument("--output-dir", default="o6_deadman_open_settle_reports")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg

AXES = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
LEFT_JOINTS = tuple("lh_" + axis for axis in AXES)
RIGHT_JOINTS = tuple("rh_" + axis for axis in AXES)
JOINTS = LEFT_JOINTS + RIGHT_JOINTS

PHYSICS_DT = 1.0 / 60.0
OPEN_WARMUP_S = 1.0
NONOPEN_TRACKING_S = 3.0
STALE_TIMEOUT_S = 0.250
MAX_OPEN_SETTLE_S = 5.0
NONOPEN_NORMALIZED = 0.65
POSITION_TOLERANCE_RAD = 0.03
VELOCITY_TOLERANCE_RAD_S = 0.10
REQUIRED_STABLE_STEPS = 6


def make_hand_cfg(urdf_path, prefix, position):
    joint_names = tuple(prefix + axis for axis in AXES)
    return ArticulationCfg(
        prim_path="/World/" + ("LeftHand" if prefix == "lh_" else "RightHand"),
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            usd_dir="/tmp/o6_usd/" + prefix[:2],
            fix_base=True,
            merge_fixed_joints=True,
            convert_mimic_joints_to_normal_joints=True,
            self_collision=False,
            make_instanceable=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=30.0,
                    damping=1.0,
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=position),
        actuators={
            "o6": ImplicitActuatorCfg(
                joint_names_expr=list(joint_names),
                effort_limit_sim=10.0,
                velocity_limit_sim=3.0,
                stiffness=30.0,
                damping=1.0,
            )
        },
    )


def set_joint_targets(hand, joint_ids, values):
    target = hand.data.default_joint_pos.clone()
    target[:, joint_ids] = torch.as_tensor(values, device=target.device, dtype=target.dtype).unsqueeze(0)
    hand.set_joint_position_target(target)


def event(sim_time_s, wall_monotonic_s):
    return {"sim_time_s": float(sim_time_s), "wall_monotonic_s": float(wall_monotonic_s)}


def write_csv(path, samples):
    fieldnames = [
        "sample_index",
        "sim_time_s",
        "wall_monotonic_s",
        "phase",
        "source_active",
        "stale_detected",
        "open_target_written",
        "global_stable_streak_steps",
    ]
    for joint in JOINTS:
        fieldnames.extend(
            [
                joint + "_q_cmd_rad",
                joint + "_q_open_rad",
                joint + "_q_actual_rad",
                joint + "_qd_actual_rad_s",
                joint + "_open_abs_error_rad",
                joint + "_within_settle_band",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = {
                "sample_index": sample["sample_index"],
                "sim_time_s": f'{sample["sim_time_s"]:.9f}',
                "wall_monotonic_s": f'{sample["wall_monotonic_s"]:.9f}',
                "phase": sample["phase"],
                "source_active": int(sample["source_active"]),
                "stale_detected": int(sample["stale_detected"]),
                "open_target_written": int(sample["open_target_written"]),
                "global_stable_streak_steps": sample["global_stable_streak_steps"],
            }
            for index, joint in enumerate(JOINTS):
                row[joint + "_q_cmd_rad"] = f'{sample["q_cmd"][index]:.9f}'
                row[joint + "_q_open_rad"] = f'{sample["q_open"][index]:.9f}'
                row[joint + "_q_actual_rad"] = f'{sample["q_actual"][index]:.9f}'
                row[joint + "_qd_actual_rad_s"] = f'{sample["qd_actual"][index]:.9f}'
                row[joint + "_open_abs_error_rad"] = f'{sample["open_abs_error"][index]:.9f}'
                row[joint + "_within_settle_band"] = int(sample["within_settle_band"][index])
            writer.writerow(row)


def write_markdown(path, report):
    timeline = report["timeline"]
    checks = report["checks"]
    lines = [
        "# Linker O6 Isaac Sim deadman open-settle evaluation",
        "",
        f'- Result: **{"PASS" if report["overall_pass"] else "FAIL"}**',
        f'- Device: `{report["simulation"]["device"]}`',
        f'- Physics rate: `{1.0 / report["simulation"]["dt_s"]:.1f} Hz`',
        f'- Stale timeout: `{report["protocol"]["stale_timeout_s"] * 1000.0:.1f} ms`',
        f'- Stable condition: all 12 joints within `+/-{report["thresholds"]["position_abs_error_rad"]:.2f} rad` and '
        f'`|qd| <= {report["thresholds"]["velocity_abs_rad_s"]:.2f} rad/s` for '
        f'`{report["thresholds"]["consecutive_steps"]}` consecutive 60 Hz steps',
        f'- Raw CSV: `{report["artifacts"]["csv"]}`',
        "",
        "## Event timeline",
        "",
        "| Event | Sim time | Wall monotonic | Latency from input stop (sim) |",
        "|---|---:|---:|---:|",
    ]
    input_stop = timeline["input_stopped"]
    for name in ("input_stopped", "stale_detected", "open_target_written", "actual_settle_confirmed"):
        item = timeline.get(name)
        if item is None:
            lines.append(f"| `{name}` | n/a | n/a | n/a |")
            continue
        latency_ms = (item["sim_time_s"] - input_stop["sim_time_s"]) * 1000.0
        lines.append(
            f'| `{name}` | {item["sim_time_s"]:.6f} s | {item["wall_monotonic_s"]:.6f} s | {latency_ms:.2f} ms |'
        )
    lines.extend(
        [
            "",
            "## Acceptance checks",
            "",
            "| Check | Result |",
            "|---|:---:|",
        ]
    )
    for name, passed in checks.items():
        lines.append(f'| {name} | {"PASS" if passed else "FAIL"} |')
    lines.extend(
        [
            "",
            "## Per-joint open settling",
            "",
            "| Joint | Settle latency from open write | Stable-window max error | Stable-window max speed | Result |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for joint, metrics in report["per_joint"].items():
        latency = metrics["settle_latency_from_open_write_s"]
        latency_text = "n/a" if latency is None else f"{latency * 1000.0:.2f} ms"
        error = metrics["stable_window_max_abs_error_rad"]
        velocity = metrics["stable_window_max_abs_velocity_rad_s"]
        error_text = "n/a" if error is None else f"{error:.6f} rad"
        velocity_text = "n/a" if velocity is None else f"{velocity:.6f} rad/s"
        lines.append(
            f'| `{joint}` | {latency_text} | {error_text} | {velocity_text} | '
            f'{"PASS" if metrics["pass"] else "FAIL"} |'
        )
    lines.extend(
        [
            "",
            "The 250 ms watchdog is driven by deterministic simulation time; wall-monotonic timestamps are diagnostic only.",
            "",
            "This is a fixed-base Isaac Sim regression only. It does not validate physical O6 motion, thumb-yaw hardware direction, DDS, or serial control.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
    if not str(sim.device).startswith("cuda"):
        raise RuntimeError(f"This evaluation requires CUDA, found device={sim.device}")
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/DomeLight", light_cfg)

    left = Articulation(make_hand_cfg(args_cli.left_urdf, "lh_", (0.0, -0.11, 0.75)))
    right = Articulation(make_hand_cfg(args_cli.right_urdf, "rh_", (0.0, 0.11, 0.75)))
    sim.set_camera_view(eye=[0.65, 0.0, 0.86], target=[0.0, 0.0, 0.86])
    sim.reset()

    left_ids, left_names = left.find_joints(LEFT_JOINTS, preserve_order=True)
    right_ids, right_names = right.find_joints(RIGHT_JOINTS, preserve_order=True)
    if tuple(left_names) != LEFT_JOINTS or tuple(right_names) != RIGHT_JOINTS:
        raise RuntimeError(f"Unexpected O6 joint order: left={left_names}, right={right_names}")

    left.reset()
    right.reset()
    left_limits = left.data.soft_joint_pos_limits[0, left_ids, :].detach().cpu().numpy()
    right_limits = right.data.soft_joint_pos_limits[0, right_ids, :].detach().cpu().numpy()
    limits = np.concatenate((left_limits, right_limits), axis=0).astype(np.float64)
    if not np.isfinite(limits).all() or np.any(limits[:, 1] <= limits[:, 0]):
        raise RuntimeError("O6 soft joint limits must be finite and ordered")
    open_target = limits[:, 0]
    nonopen_target = limits[:, 0] + NONOPEN_NORMALIZED * (limits[:, 1] - limits[:, 0])
    if np.any(np.abs(nonopen_target - open_target) <= POSITION_TOLERANCE_RAD):
        raise RuntimeError("Non-open test target must be outside the open settle band for every joint")

    input_stop_sim_time = OPEN_WARMUP_S + NONOPEN_TRACKING_S
    end_sim_time = input_stop_sim_time + STALE_TIMEOUT_S + MAX_OPEN_SETTLE_S
    sample_count = int(math.ceil(end_sim_time / PHYSICS_DT))
    samples = []
    events = {
        "input_stopped": None,
        "stale_detected": None,
        "open_target_written": None,
        "actual_settle_confirmed": None,
    }
    per_joint_streak = np.zeros(len(JOINTS), dtype=np.int64)
    per_joint_confirmed = [None] * len(JOINTS)
    global_streak = 0
    nonopen_precondition_streak = 0
    wall_started = time.monotonic()
    print(
        f"[O6 DEADMAN] device={sim.device} stale_timeout={STALE_TIMEOUT_S * 1000.0:.1f}ms "
        f"stable_steps={REQUIRED_STABLE_STEPS}"
    )

    for step_index in range(sample_count):
        command_time = step_index * PHYSICS_DT
        source_active = command_time < input_stop_sim_time
        stale = command_time - input_stop_sim_time >= STALE_TIMEOUT_S - 1e-12
        if command_time < OPEN_WARMUP_S:
            phase = "open_warmup"
            command = open_target
        elif source_active:
            phase = "nonopen_tracking"
            command = nonopen_target
        elif not stale:
            phase = "input_stopped_hold_last"
            command = nonopen_target
            if events["input_stopped"] is None:
                events["input_stopped"] = event(input_stop_sim_time, time.monotonic())
                print(f"[O6 DEADMAN] input_stopped sim={input_stop_sim_time:.6f}s")
        else:
            phase = "stale_open"
            command = open_target
            if events["stale_detected"] is None:
                events["stale_detected"] = event(command_time, time.monotonic())
                print(f"[O6 DEADMAN] stale_detected sim={command_time:.6f}s")

        set_joint_targets(left, left_ids, command[: len(AXES)])
        set_joint_targets(right, right_ids, command[len(AXES) :])
        left.write_data_to_sim()
        right.write_data_to_sim()
        if stale and events["open_target_written"] is None:
            events["open_target_written"] = event(command_time, time.monotonic())
            print(f"[O6 DEADMAN] open_target_written sim={command_time:.6f}s")
        sim.step()
        left.update(sim.get_physics_dt())
        right.update(sim.get_physics_dt())

        command_buffer = np.concatenate(
            (
                left.data.joint_pos_target[0, left_ids].detach().cpu().numpy(),
                right.data.joint_pos_target[0, right_ids].detach().cpu().numpy(),
            )
        ).astype(np.float64)
        if not np.allclose(command_buffer, command, rtol=0.0, atol=1e-6):
            raise RuntimeError("Isaac joint position target buffer does not match the requested command")
        actual = np.concatenate(
            (
                left.data.joint_pos[0, left_ids].detach().cpu().numpy(),
                right.data.joint_pos[0, right_ids].detach().cpu().numpy(),
            )
        ).astype(np.float64)
        actual_velocity = np.concatenate(
            (
                left.data.joint_vel[0, left_ids].detach().cpu().numpy(),
                right.data.joint_vel[0, right_ids].detach().cpu().numpy(),
            )
        ).astype(np.float64)
        sample_sim_time = command_time + PHYSICS_DT
        open_abs_error = np.abs(actual - open_target)
        within = (open_abs_error <= POSITION_TOLERANCE_RAD) & (
            np.abs(actual_velocity) <= VELOCITY_TOLERANCE_RAD_S
        )

        if phase == "nonopen_tracking":
            nonopen_within = (np.abs(actual - nonopen_target) <= POSITION_TOLERANCE_RAD) & (
                np.abs(actual_velocity) <= VELOCITY_TOLERANCE_RAD_S
            )
            nonopen_precondition_streak = nonopen_precondition_streak + 1 if np.all(nonopen_within) else 0
        if events["open_target_written"] is not None:
            per_joint_streak = np.where(within, per_joint_streak + 1, 0)
            for joint_index in range(len(JOINTS)):
                if per_joint_confirmed[joint_index] is None and per_joint_streak[joint_index] >= REQUIRED_STABLE_STEPS:
                    per_joint_confirmed[joint_index] = event(sample_sim_time, time.monotonic())
            global_streak = global_streak + 1 if np.all(within) else 0
            if global_streak >= REQUIRED_STABLE_STEPS and events["actual_settle_confirmed"] is None:
                events["actual_settle_confirmed"] = event(sample_sim_time, time.monotonic())
                print(f"[O6 DEADMAN] actual_settle_confirmed sim={sample_sim_time:.6f}s")

        samples.append(
            {
                "sample_index": step_index,
                "sim_time_s": sample_sim_time,
                "wall_monotonic_s": time.monotonic(),
                "phase": phase,
                "source_active": source_active,
                "stale_detected": events["stale_detected"] is not None,
                "open_target_written": events["open_target_written"] is not None,
                "global_stable_streak_steps": global_streak,
                "q_cmd": command_buffer,
                "q_open": open_target,
                "q_actual": actual,
                "qd_actual": actual_velocity,
                "open_abs_error": open_abs_error,
                "within_settle_band": within,
            }
        )
        if events["actual_settle_confirmed"] is not None:
            break

    final_sample = samples[-1]
    stable_window = samples[-REQUIRED_STABLE_STEPS:] if events["actual_settle_confirmed"] is not None else []
    open_written = events["open_target_written"]
    per_joint = {}
    for index, joint in enumerate(JOINTS):
        confirmed = per_joint_confirmed[index]
        stable_window_errors = [sample["open_abs_error"][index] for sample in stable_window]
        stable_window_velocities = [abs(sample["qd_actual"][index]) for sample in stable_window]
        per_joint[joint] = {
            "settle_confirmed": confirmed,
            "settle_latency_from_open_write_s": (
                None if confirmed is None else confirmed["sim_time_s"] - open_written["sim_time_s"]
            ),
            "final_abs_error_rad": float(final_sample["open_abs_error"][index]),
            "final_abs_velocity_rad_s": float(abs(final_sample["qd_actual"][index])),
            "stable_window_max_abs_error_rad": (
                None if not stable_window_errors else float(max(stable_window_errors))
            ),
            "stable_window_max_abs_velocity_rad_s": (
                None if not stable_window_velocities else float(max(stable_window_velocities))
            ),
            "pass": confirmed is not None,
        }

    stale_event = events["stale_detected"]
    settle_event = events["actual_settle_confirmed"]
    stale_delay = None if stale_event is None else stale_event["sim_time_s"] - input_stop_sim_time
    checks = {
        "non-open pose was stable before input stopped": nonopen_precondition_streak >= REQUIRED_STABLE_STEPS,
        "stale detected at 250 ms simulation time within one physics step": (
            stale_delay is not None
            and STALE_TIMEOUT_S <= stale_delay <= STALE_TIMEOUT_S + PHYSICS_DT + 1e-9
        ),
        "open target written on stale-detection step": (
            stale_event is not None
            and open_written is not None
            and abs(stale_event["sim_time_s"] - open_written["sim_time_s"]) <= 1e-12
        ),
        "open command buffer matched lower joint limits": bool(
            np.allclose(final_sample["q_cmd"], open_target, rtol=0.0, atol=1e-6)
        ),
        "all joints settled for six consecutive steps": settle_event is not None,
        "each joint reached the settle condition": all(item["pass"] for item in per_joint.values()),
        "final position errors satisfy threshold": bool(
            np.all(final_sample["open_abs_error"] <= POSITION_TOLERANCE_RAD)
        ),
        "final joint speeds satisfy threshold": bool(
            np.all(np.abs(final_sample["qd_actual"]) <= VELOCITY_TOLERANCE_RAD_S)
        ),
    }

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "o6_deadman_open_settle_" + run_id
    csv_path = output_dir / (stem + ".csv")
    json_path = output_dir / (stem + ".json")
    markdown_path = output_dir / (stem + ".md")
    write_csv(csv_path, samples)

    report = {
        "schema": "linker_o6_deadman_open_settle_evaluation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(checks.values()),
        "simulation": {
            "expected_runtime": "Isaac Sim 5.1",
            "device": str(sim.device),
            "dt_s": PHYSICS_DT,
            "wall_duration_s": time.monotonic() - wall_started,
            "fixed_base": True,
            "actuator": {
                "stiffness": 30.0,
                "damping": 1.0,
                "effort_limit_sim": 10.0,
                "velocity_limit_sim": 3.0,
            },
            "left_urdf": args_cli.left_urdf,
            "right_urdf": args_cli.right_urdf,
        },
        "protocol": {
            "open_warmup_s": OPEN_WARMUP_S,
            "nonopen_tracking_s": NONOPEN_TRACKING_S,
            "nonopen_normalized": NONOPEN_NORMALIZED,
            "stale_timeout_s": STALE_TIMEOUT_S,
            "maximum_open_settle_s": MAX_OPEN_SETTLE_S,
            "stale_is_simulated": True,
        },
        "thresholds": {
            "position_abs_error_rad": POSITION_TOLERANCE_RAD,
            "velocity_abs_rad_s": VELOCITY_TOLERANCE_RAD_S,
            "consecutive_steps": REQUIRED_STABLE_STEPS,
        },
        "joint_order": list(JOINTS),
        "open_target_rad": {joint: float(open_target[index]) for index, joint in enumerate(JOINTS)},
        "nonopen_target_rad": {joint: float(nonopen_target[index]) for index, joint in enumerate(JOINTS)},
        "timeline": events,
        "latency": {
            "input_stop_to_stale_detected_sim_s": stale_delay,
            "input_stop_to_stale_detected_wall_s": (
                None
                if stale_event is None
                else stale_event["wall_monotonic_s"] - events["input_stopped"]["wall_monotonic_s"]
            ),
            "stale_detected_to_open_target_written_sim_s": (
                None
                if stale_event is None or open_written is None
                else open_written["sim_time_s"] - stale_event["sim_time_s"]
            ),
            "stale_detected_to_open_target_written_wall_s": (
                None
                if stale_event is None or open_written is None
                else open_written["wall_monotonic_s"] - stale_event["wall_monotonic_s"]
            ),
            "open_target_written_to_actual_settle_confirmed_sim_s": (
                None
                if open_written is None or settle_event is None
                else settle_event["sim_time_s"] - open_written["sim_time_s"]
            ),
            "open_target_written_to_actual_settle_confirmed_wall_s": (
                None
                if open_written is None or settle_event is None
                else settle_event["wall_monotonic_s"] - open_written["wall_monotonic_s"]
            ),
        },
        "per_joint": per_joint,
        "checks": checks,
        "artifacts": {
            "sample_count": len(samples),
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(markdown_path),
        },
        "safety": (
            "Fixed-base Isaac Sim only; no DDS, network, serial, device node, or physical O6 command path is used."
        ),
    }
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(markdown_path, report)
    print(f'[O6 DEADMAN] result={"PASS" if report["overall_pass"] else "FAIL"}')
    print(f"[O6 DEADMAN] csv={csv_path}")
    print(f"[O6 DEADMAN] json={json_path}")
    print(f"[O6 DEADMAN] markdown={markdown_path}")
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
