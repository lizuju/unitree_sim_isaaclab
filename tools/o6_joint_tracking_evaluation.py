import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Linker O6 joint tracking in Isaac Sim.")
parser.add_argument("--left-urdf", default="/home/code/linkerhand-urdf/O6/left/linkerhand_o6_left.urdf")
parser.add_argument("--right-urdf", default="/home/code/linkerhand-urdf/O6/right/linkerhand_o6_right.urdf")
parser.add_argument("--output-dir", default="o6_joint_tracking_reports")
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
STEP_RAD = 0.10
SINE_AMPLITUDE_RAD = 0.08
SETTLE_BAND_RAD = 0.01

THRESHOLDS = {
    "position_rmse_rad": 0.040,
    "position_p95_abs_rad": 0.075,
    "position_max_abs_rad": 0.120,
    "sine_velocity_rmse_rad_s": 0.600,
    "sine_velocity_p95_abs_rad_s": 1.200,
    "hold_jitter_std_rad": 0.003,
    "hold_jitter_peak_to_peak_rad": 0.015,
    "step_settle_time_s": 1.500,
    "step_overshoot_rad": 0.040,
    "step_final_abs_error_rad": 0.020,
}


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


def make_protocol():
    phases = [
        {"name": "warmup", "kind": "hold", "duration_s": 1.5},
        {"name": "hold", "kind": "hold", "duration_s": 2.0},
    ]
    for axis_index, axis in enumerate(AXES):
        phases.append(
            {
                "name": "step_" + axis,
                "kind": "step",
                "axis_index": axis_index,
                "duration_s": 2.0,
            }
        )
        phases.append({"name": "reset_" + axis, "kind": "hold", "duration_s": 0.75})
    phases.extend(
        [
            {"name": "sine_0p2hz", "kind": "sine", "frequency_hz": 0.2, "duration_s": 10.0},
            {"name": "reset_after_0p2hz", "kind": "hold", "duration_s": 1.0},
            {"name": "sine_0p5hz", "kind": "sine", "frequency_hz": 0.5, "duration_s": 6.0},
            {"name": "final_hold", "kind": "hold", "duration_s": 1.0},
        ]
    )
    start = 0.0
    for phase in phases:
        phase["start_s"] = start
        start += phase["duration_s"]
        phase["end_s"] = start
    return phases, start


def command_for_phase(phase, phase_elapsed, base):
    command = base.copy()
    command_velocity = np.zeros_like(base)
    if phase["kind"] == "step":
        axis = phase["axis_index"]
        command[[axis, axis + len(AXES)]] += STEP_RAD
    elif phase["kind"] == "sine":
        frequency = phase["frequency_hz"]
        angle = 2.0 * math.pi * frequency * phase_elapsed
        command += SINE_AMPLITUDE_RAD * math.sin(angle)
        command_velocity += SINE_AMPLITUDE_RAD * 2.0 * math.pi * frequency * math.cos(angle)
    return command, command_velocity


def scalar_error_metrics(error):
    absolute = np.abs(error)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def step_metrics(actual, command, phase_elapsed):
    error = actual - command
    absolute = np.abs(error)
    tail_max = np.maximum.accumulate(absolute[::-1])[::-1]
    settled = np.flatnonzero(tail_max <= SETTLE_BAND_RAD)
    settle_time = float(phase_elapsed[settled[0]]) if settled.size else None
    return {
        "settle_band_rad": SETTLE_BAND_RAD,
        "settle_time_s": settle_time,
        "overshoot_rad": float(max(0.0, np.max(error))),
        "final_abs_error_rad": float(absolute[-1]),
    }


def compute_summary(samples, base):
    phases = np.asarray([sample["phase"] for sample in samples])
    phase_elapsed = np.asarray([sample["phase_elapsed_s"] for sample in samples], dtype=np.float64)
    command = np.asarray([sample["q_cmd"] for sample in samples], dtype=np.float64)
    actual = np.asarray([sample["q_actual"] for sample in samples], dtype=np.float64)
    command_velocity = np.asarray([sample["qd_cmd"] for sample in samples], dtype=np.float64)
    actual_velocity = np.asarray([sample["qd_actual"] for sample in samples], dtype=np.float64)

    evaluation_mask = np.asarray(
        [phase == "hold" or phase.startswith("step_") or phase.startswith("sine_") for phase in phases]
    )
    hold_mask = (phases == "hold") & (phase_elapsed >= 0.5)
    sine_mask = np.asarray([phase.startswith("sine_") for phase in phases])
    sine_frequency_masks = {
        "0.2": phases == "sine_0p2hz",
        "0.5": phases == "sine_0p5hz",
    }

    per_joint = {}
    for joint_index, joint_name in enumerate(JOINTS):
        position_error = actual[evaluation_mask, joint_index] - command[evaluation_mask, joint_index]
        position = scalar_error_metrics(position_error)
        velocity_error = actual_velocity[sine_mask, joint_index] - command_velocity[sine_mask, joint_index]
        velocity = scalar_error_metrics(velocity_error)
        hold_values = actual[hold_mask, joint_index]
        hold_jitter = {
            "std_rad": float(np.std(hold_values)),
            "peak_to_peak_rad": float(np.ptp(hold_values)),
            "mean_abs_error_rad": float(np.mean(np.abs(hold_values - base[joint_index]))),
        }

        axis = AXES[joint_index % len(AXES)]
        step_mask = phases == "step_" + axis
        step = step_metrics(
            actual[step_mask, joint_index],
            command[step_mask, joint_index],
            phase_elapsed[step_mask],
        )

        sine = {}
        for frequency, mask in sine_frequency_masks.items():
            sine[frequency] = {
                "position": scalar_error_metrics(actual[mask, joint_index] - command[mask, joint_index]),
                "velocity": scalar_error_metrics(
                    actual_velocity[mask, joint_index] - command_velocity[mask, joint_index]
                ),
            }

        checks = {
            "position_rmse": position["rmse"] <= THRESHOLDS["position_rmse_rad"],
            "position_p95": position["p95_abs"] <= THRESHOLDS["position_p95_abs_rad"],
            "position_max": position["max_abs"] <= THRESHOLDS["position_max_abs_rad"],
            "sine_velocity_rmse": velocity["rmse"] <= THRESHOLDS["sine_velocity_rmse_rad_s"],
            "sine_velocity_p95": velocity["p95_abs"] <= THRESHOLDS["sine_velocity_p95_abs_rad_s"],
            "hold_jitter_std": hold_jitter["std_rad"] <= THRESHOLDS["hold_jitter_std_rad"],
            "hold_jitter_peak_to_peak": (
                hold_jitter["peak_to_peak_rad"] <= THRESHOLDS["hold_jitter_peak_to_peak_rad"]
            ),
            "step_settle": (
                step["settle_time_s"] is not None
                and step["settle_time_s"] <= THRESHOLDS["step_settle_time_s"]
            ),
            "step_overshoot": step["overshoot_rad"] <= THRESHOLDS["step_overshoot_rad"],
            "step_final_error": (
                step["final_abs_error_rad"] <= THRESHOLDS["step_final_abs_error_rad"]
            ),
        }
        per_joint[joint_name] = {
            "position_error_rad": position,
            "sine_velocity_error_rad_s": velocity,
            "hold_jitter": hold_jitter,
            "paired_step": step,
            "sine_by_frequency_hz": sine,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return per_joint


def write_csv(path, samples):
    fieldnames = ["sim_time_s", "wall_monotonic_s", "phase", "phase_elapsed_s"]
    for joint in JOINTS:
        fieldnames.extend(
            [
                joint + "_q_cmd_rad",
                joint + "_q_actual_rad",
                joint + "_qd_cmd_rad_s",
                joint + "_qd_actual_rad_s",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = {
                "sim_time_s": f'{sample["sim_time_s"]:.9f}',
                "wall_monotonic_s": f'{sample["wall_monotonic_s"]:.9f}',
                "phase": sample["phase"],
                "phase_elapsed_s": f'{sample["phase_elapsed_s"]:.9f}',
            }
            for index, joint in enumerate(JOINTS):
                row[joint + "_q_cmd_rad"] = f'{sample["q_cmd"][index]:.9f}'
                row[joint + "_q_actual_rad"] = f'{sample["q_actual"][index]:.9f}'
                row[joint + "_qd_cmd_rad_s"] = f'{sample["qd_cmd"][index]:.9f}'
                row[joint + "_qd_actual_rad_s"] = f'{sample["qd_actual"][index]:.9f}'
            writer.writerow(row)


def write_markdown(path, report):
    lines = [
        "# Linker O6 Isaac Sim joint tracking evaluation",
        "",
        f'- Result: **{"PASS" if report["overall_pass"] else "FAIL"}**',
        f'- Device: `{report["simulation"]["device"]}`',
        f'- Physics rate: {1.0 / report["simulation"]["dt_s"]:.1f} Hz',
        f'- Simulated duration: {report["simulation"]["duration_s"]:.2f} s',
        f'- Samples: {report["artifacts"]["sample_count"]}',
        f'- Raw CSV: `{report["artifacts"]["csv"]}`',
        "",
        "The paired step applies +0.10 rad to the same axis on both hands. Velocity error is measured only",
        "during the smooth 0.2 Hz and 0.5 Hz sine phases; hold jitter excludes the first 0.5 s.",
        "",
        "## Per-joint results",
        "",
        "| Joint | Pos RMSE | Pos p95 | Pos max | Vel RMSE | Vel p95 | Hold std | Hold p-p | Settle | Overshoot | Final err | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for joint, metrics in report["per_joint"].items():
        position = metrics["position_error_rad"]
        velocity = metrics["sine_velocity_error_rad_s"]
        jitter = metrics["hold_jitter"]
        step = metrics["paired_step"]
        settle = "n/a" if step["settle_time_s"] is None else f'{step["settle_time_s"]:.3f}s'
        lines.append(
            f'| `{joint}` | {position["rmse"]:.4f} | {position["p95_abs"]:.4f} | '
            f'{position["max_abs"]:.4f} | {velocity["rmse"]:.4f} | {velocity["p95_abs"]:.4f} | '
            f'{jitter["std_rad"]:.5f} | {jitter["peak_to_peak_rad"]:.5f} | {settle} | '
            f'{step["overshoot_rad"]:.4f} | {step["final_abs_error_rad"]:.4f} | '
            f'{"PASS" if metrics["pass"] else "FAIL"} |'
        )

    lines.extend(["", "## PASS thresholds", ""])
    for name, value in report["thresholds"].items():
        lines.append(f'- `{name}`: {value}')
    lines.extend(
        [
            "",
            "A failed threshold is a simulation regression signal, not a physical-hand safety certification.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
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
    base = np.mean(limits, axis=1)
    if not np.isfinite(limits).all():
        raise RuntimeError("O6 soft joint limits must be finite")
    if np.any(base + STEP_RAD > limits[:, 1]) or np.any(base - SINE_AMPLITUDE_RAD < limits[:, 0]):
        raise RuntimeError("O6 joint limits are too narrow for the evaluation trajectory")
    if np.any(base + SINE_AMPLITUDE_RAD > limits[:, 1]):
        raise RuntimeError("O6 joint limits are too narrow for the sine trajectory")

    protocol, duration = make_protocol()
    sample_count = int(math.ceil(duration / PHYSICS_DT))
    samples = []
    phase_index = 0
    wall_started = time.monotonic()
    print(f"[O6 TRACKING] device={sim.device} duration={duration:.2f}s samples={sample_count}")

    for step_index in range(sample_count):
        sim_time = step_index * PHYSICS_DT
        while phase_index + 1 < len(protocol) and sim_time >= protocol[phase_index]["end_s"]:
            phase_index += 1
            print(f'[O6 TRACKING] phase={protocol[phase_index]["name"]}')
        phase = protocol[phase_index]
        phase_elapsed = sim_time - phase["start_s"]
        command, command_velocity = command_for_phase(phase, phase_elapsed, base)

        set_joint_targets(left, left_ids, command[: len(AXES)])
        set_joint_targets(right, right_ids, command[len(AXES) :])
        left.write_data_to_sim()
        right.write_data_to_sim()
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
        samples.append(
            {
                "sim_time_s": sim_time,
                "wall_monotonic_s": time.monotonic(),
                "phase": phase["name"],
                "phase_elapsed_s": phase_elapsed,
                "q_cmd": command_buffer,
                "q_actual": actual,
                "qd_cmd": command_velocity,
                "qd_actual": actual_velocity,
            }
        )

    per_joint = compute_summary(samples, base)
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "o6_joint_tracking_" + run_id
    csv_path = output_dir / (stem + ".csv")
    json_path = output_dir / (stem + ".json")
    markdown_path = output_dir / (stem + ".md")
    write_csv(csv_path, samples)

    report = {
        "schema": "linker_o6_joint_tracking_evaluation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(metrics["pass"] for metrics in per_joint.values()),
        "simulation": {
            "device": str(sim.device),
            "dt_s": PHYSICS_DT,
            "duration_s": duration,
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
            "paired_step_rad": STEP_RAD,
            "sine_amplitude_rad": SINE_AMPLITUDE_RAD,
            "sine_frequencies_hz": [0.2, 0.5],
            "phases": protocol,
        },
        "joint_order": list(JOINTS),
        "base_position_rad": {joint: float(base[index]) for index, joint in enumerate(JOINTS)},
        "thresholds": THRESHOLDS,
        "per_joint": per_joint,
        "artifacts": {
            "sample_count": len(samples),
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(markdown_path),
        },
    }
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(markdown_path, report)
    print(f'[O6 TRACKING] result={"PASS" if report["overall_pass"] else "FAIL"}')
    print(f"[O6 TRACKING] csv={csv_path}")
    print(f"[O6 TRACKING] json={json_path}")
    print(f"[O6 TRACKING] markdown={markdown_path}")
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
