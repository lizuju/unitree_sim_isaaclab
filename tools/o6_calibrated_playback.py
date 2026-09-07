import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay calibrated Vision Pro targets on two Linker O6 hands.")
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument("--recording")
source_group.add_argument("--live-state")
parser.add_argument("--live-timeout", type=float, default=0.25)
parser.add_argument("--expected-mapping")
parser.add_argument("--left-urdf", default="/home/code/linkerhand-urdf/O6/left/linkerhand_o6_left.urdf")
parser.add_argument("--right-urdf", default="/home/code/linkerhand-urdf/O6/right/linkerhand_o6_right.urdf")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import bisect
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
LEFT_THUMB_NORMALIZED_MAX = 1.08 / (2.29 * 0.58)
LIVE_AXIS_ORDER = ["thumb_pitch", "thumb_yaw", "index", "middle", "ring", "pinky"]


def load_playback(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    frames_by_timestamp = {}
    for row in rows:
        if not row.get("armed") or "target_12" not in row:
            continue
        if not row.get("visionpro_input_calibrated"):
            raise ValueError("Recording is not Vision Pro input calibrated")
        if row.get("thumb_yaw_hardware_direction_validated") is not False:
            raise ValueError("Recording must remain hardware-direction unvalidated")
        frames_by_timestamp[row["monotonic_timestamp"]] = row["target_12"]

    timestamps = sorted(frames_by_timestamp)
    if len(timestamps) < 2:
        raise ValueError("Recording does not contain enough calibrated frames")
    targets = np.asarray([frames_by_timestamp[timestamp] for timestamp in timestamps], dtype=np.float32)
    if targets.shape[1] != 12 or not np.isfinite(targets).all():
        raise ValueError("Each calibrated frame must contain 12 finite targets")
    targets[:, 0] = np.minimum(targets[:, 0], LEFT_THUMB_NORMALIZED_MAX)
    relative_time = np.asarray(timestamps, dtype=np.float64) - timestamps[0]
    return relative_time, targets


def load_live_frame(path, now, timeout, expected_mapping):
    try:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        sequence = row.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
            return None, "invalid_sequence", None, None, None
        if row.get("armed") is not True:
            status = row.get("reason", "disarmed") if row.get("armed") is False else "invalid_armed"
            return None, status, sequence, None, None
        if row.get("schema") != "linker_o6_target_v1":
            return None, "schema_rejected", sequence, None, None
        if row.get("mapping") != expected_mapping:
            return None, "mapping_rejected", sequence, None, None
        if row.get("visionpro_input_calibrated") is not True:
            return None, "uncalibrated", sequence, None, None
        if row.get("thumb_yaw_hardware_direction_validated") is not False:
            return None, "hardware_direction_rejected", sequence, None, None
        if row.get("target_units") != "normalized_0_1":
            return None, "units_rejected", sequence, None, None
        if row.get("hardware_axis_order") != LIVE_AXIS_ORDER:
            return None, "axis_order_rejected", sequence, None, None
        if row.get("target_hand_order") != ["left", "right"]:
            return None, "hand_order_rejected", sequence, None, None

        published_ns = row.get("published_monotonic_ns")
        if not isinstance(published_ns, int) or isinstance(published_ns, bool) or published_ns <= 0:
            return None, "invalid_published_timestamp", sequence, None, None
        source_value = row.get("monotonic_timestamp")
        if not isinstance(source_value, (int, float)) or isinstance(source_value, bool):
            return None, "invalid_source_timestamp", sequence, None, None
        published = published_ns / 1_000_000_000.0
        source_timestamp = float(source_value)
        if not math.isfinite(source_timestamp):
            return None, "invalid_source_timestamp", sequence, None, None
        transport_age = now - published
        tracking_age = now - source_timestamp
        if not (0.0 <= transport_age <= timeout and 0.0 <= tracking_age <= timeout):
            return None, "stale", sequence, source_timestamp, tracking_age

        frame = np.asarray(row["target_12"], dtype=np.float64)
        if frame.shape != (12,) or not np.isfinite(frame).all():
            return None, "invalid_target", sequence, source_timestamp, tracking_age
        if np.any(frame < 0.0) or np.any(frame > 1.0):
            return None, "target_out_of_range", sequence, source_timestamp, tracking_age
        frame = frame.astype(np.float32)
        frame[0] = min(frame[0], LEFT_THUMB_NORMALIZED_MAX)
        return frame, "armed", sequence, source_timestamp, tracking_age
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError):
        return None, "unavailable", None, None, None


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


def command_hand(hand, joint_ids, normalized):
    target = hand.data.default_joint_pos.clone()
    limits = hand.data.soft_joint_pos_limits[:, joint_ids, :]
    values = torch.tensor(normalized, device=target.device, dtype=target.dtype).unsqueeze(0)
    target[:, joint_ids] = limits[..., 0] + values * (limits[..., 1] - limits[..., 0])
    hand.set_joint_position_target(target)


def main():
    if not (math.isfinite(args_cli.live_timeout) and args_cli.live_timeout > 0.0):
        raise ValueError("--live-timeout must be positive")
    if args_cli.live_state and not args_cli.expected_mapping:
        raise ValueError("--expected-mapping is required with --live-state")
    if args_cli.recording:
        playback_time, targets = load_playback(args_cli.recording)
    else:
        playback_time = None
        targets = None

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device))
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
    started = time.monotonic()
    print(f"[O6 PREVIEW] left thumb clamp={LEFT_THUMB_NORMALIZED_MAX:.6f}; DDS and serial are not used")
    if args_cli.recording:
        duration = float(playback_time[-1])
        completed_loops = -1
        print(f"[O6 PREVIEW] frames={len(targets)} duration={duration:.3f}s device={sim.device}")
    else:
        live_state_path = Path(args_cli.live_state)
        frame = np.zeros(12, dtype=np.float32)
        live_status = None
        last_sequence = None
        last_source_timestamp = None
        report_started = started
        sim_steps = 0
        publish_samples = 0
        source_samples = 0
        missed_sequences = 0
        stale_steps = 0
        invalid_steps = 0
        age_samples = []
        print(f"[O6 LIVE] source={live_state_path} timeout={args_cli.live_timeout:.3f}s device={sim.device}")

    while simulation_app.is_running():
        now = time.monotonic()
        if args_cli.recording:
            elapsed = now - started
            loop_index = int(elapsed // duration)
            if loop_index != completed_loops:
                completed_loops = loop_index
                print(f"[O6 PREVIEW] playback loop {loop_index + 1}")
            cycle_time = elapsed % duration
            frame_index = max(0, bisect.bisect_right(playback_time, cycle_time) - 1)
            frame = targets[frame_index]
        else:
            live_frame, status, sequence, source_timestamp, tracking_age = load_live_frame(
                live_state_path, now, args_cli.live_timeout, args_cli.expected_mapping
            )
            if status != live_status:
                output_state = "tracking" if live_frame is not None else "open"
                print(f"[O6 LIVE] state={status} output={output_state}")
                live_status = status
            if live_frame is None:
                frame = np.zeros(12, dtype=np.float32)
                stale_steps += 1
                if status not in ("unavailable", "awaiting_r_key", "tracking_stale", "stale", "disarmed"):
                    invalid_steps += 1
            else:
                frame = live_frame
                if sequence != last_sequence:
                    publish_samples += 1
                    if last_sequence is not None and sequence > last_sequence + 1:
                        missed_sequences += sequence - last_sequence - 1
                    last_sequence = sequence
                if source_timestamp != last_source_timestamp:
                    source_samples += 1
                    last_source_timestamp = source_timestamp
                age_samples.append(tracking_age)

            sim_steps += 1
            report_elapsed = now - report_started
            if report_elapsed >= 5.0:
                if age_samples:
                    age_ms = np.asarray(age_samples) * 1000.0
                    age_summary = f"age_ms_p50={np.percentile(age_ms, 50):.1f} p95={np.percentile(age_ms, 95):.1f} max={age_ms.max():.1f}"
                else:
                    age_summary = "age_ms_p50=n/a p95=n/a max=n/a"
                print(
                    f"[O6 LIVE] sim_hz={sim_steps / report_elapsed:.1f} "
                    f"publish_hz={publish_samples / report_elapsed:.1f} "
                    f"source_hz={source_samples / report_elapsed:.1f} {age_summary} "
                    f"missed_seq={missed_sequences} stale_steps={stale_steps} invalid_steps={invalid_steps}"
                )
                report_started = now
                sim_steps = 0
                publish_samples = 0
                source_samples = 0
                missed_sequences = 0
                stale_steps = 0
                invalid_steps = 0
                age_samples.clear()

        command_hand(left, left_ids, frame[:6])
        command_hand(right, right_ids, frame[6:])
        left.write_data_to_sim()
        right.write_data_to_sim()
        sim.step()
        left.update(sim.get_physics_dt())
        right.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
