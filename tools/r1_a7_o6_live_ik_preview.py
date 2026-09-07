import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview fixed-base R1-A7 + Linker O6 with live hand and arm IK targets.")
parser.add_argument("--combined-urdf", default="/home/code/r1_a7_linker_o6.urdf")
parser.add_argument("--hand-live-state", default="/home/code/o6-live/target.json")
parser.add_argument("--arm-live-state", default="/home/code/r1-arm-live/target.json")
parser.add_argument("--live-timeout", type=float, default=0.25)
parser.add_argument("--expected-hand-mapping", default="visionpro_giovanni_20260831_v1")
parser.add_argument(
    "--expected-arm-mapping",
    default="r1_a7_keyboard_cartesian_v1,r1_a7_visionpro_cartesian_v1",
)
parser.add_argument("--max-self-contact-force", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from r1_a7_workspace import is_dual_offset_in_workspace

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import subtract_frame_transforms


PHYSICS_DT = 1.0 / 60.0
READY_SETTLE_STEPS = 240
MAX_ARM_JOINT_STEP_RAD = 0.005
JOINT_LIMIT_MARGIN_RAD = 0.10
MAX_CARTESIAN_SPEED_M_S = 0.02
MIN_JACOBIAN_SINGULAR_VALUE = 0.005
MAX_JACOBIAN_CONDITION = 500.0
MAX_ARM_SPEED_RAD_S = 0.35
IK_SETTLE_POSITION_ERROR_M = 0.001
DLS_LAMBDA = 0.05
NULLSPACE_POSTURE_GAIN = 0.10
LEFT_THUMB_NORMALIZED_MAX = 1.08 / (2.29 * 0.58)
HAND_SCHEMA = "linker_o6_target_v1"
ARM_SCHEMA = "r1_a7_arm_ik_target_v1"
HAND_AXIS_ORDER = ["thumb_pitch", "thumb_yaw", "index", "middle", "ring", "pinky"]
HAND_ORDER = ["left", "right"]
ARM_COMMAND_WAKE_THRESHOLD_M = 0.002
ACTUAL_WORKSPACE_TOLERANCE_M = 0.003

R1_JOINTS = (
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "head_pitch_joint",
    "head_yaw_joint",
)
LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS = tuple(name.replace("left_", "right_") for name in LEFT_ARM_JOINTS)
HAND_JOINTS = tuple("lh_" + name for name in (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)) + tuple("rh_" + name for name in (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
))
O6_MIMIC_JOINTS = tuple("lh_" + name for name in (
    "thumb_ip",
    "index_dip",
    "middle_dip",
    "ring_dip",
    "pinky_dip",
)) + tuple("rh_" + name for name in (
    "thumb_ip",
    "index_dip",
    "middle_dip",
    "ring_dip",
    "pinky_dip",
))
READY_ARM = {
    "left_shoulder_pitch_joint": -0.45,
    "left_shoulder_roll_joint": 0.45,
    "left_elbow_joint": 0.85,
    "right_shoulder_pitch_joint": -0.45,
    "right_shoulder_roll_joint": -0.45,
    "right_elbow_joint": 0.85,
}


def validate_common(row, now, timeout, schema, mapping):
    sequence = row.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
        return "invalid_sequence", None, None
    if row.get("armed") is not True:
        status = row.get("reason", "disarmed") if row.get("armed") is False else "invalid_armed"
        return status, sequence, None
    if row.get("schema") != schema:
        return "schema_rejected", sequence, None
    allowed_mappings = mapping.split(",") if isinstance(mapping, str) else mapping
    if row.get("mapping") not in allowed_mappings:
        return "mapping_rejected", sequence, None
    published_ns = row.get("published_monotonic_ns")
    if not isinstance(published_ns, int) or isinstance(published_ns, bool) or published_ns <= 0:
        return "invalid_published_timestamp", sequence, None
    source_value = row.get("monotonic_timestamp")
    if not isinstance(source_value, (int, float)) or isinstance(source_value, bool):
        return "invalid_source_timestamp", sequence, None
    source_timestamp = float(source_value)
    if not math.isfinite(source_timestamp):
        return "invalid_source_timestamp", sequence, None
    transport_age = now - published_ns / 1_000_000_000.0
    tracking_age = now - source_timestamp
    if not (0.0 <= transport_age <= timeout and 0.0 <= tracking_age <= timeout):
        return "stale", sequence, tracking_age
    return "armed", sequence, tracking_age


def load_hand_frame(path, now, timeout, expected_mapping):
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        status, sequence, tracking_age = validate_common(
            row, now, timeout, HAND_SCHEMA, expected_mapping
        )
        if status != "armed":
            return None, status, sequence, tracking_age
        if row.get("visionpro_input_calibrated") is not True:
            return None, "uncalibrated", sequence, tracking_age
        if row.get("thumb_yaw_hardware_direction_validated") is not False:
            return None, "hardware_direction_rejected", sequence, tracking_age
        if row.get("target_units") != "normalized_0_1":
            return None, "units_rejected", sequence, tracking_age
        if row.get("hardware_axis_order") != HAND_AXIS_ORDER:
            return None, "axis_order_rejected", sequence, tracking_age
        if row.get("target_hand_order") != HAND_ORDER:
            return None, "hand_order_rejected", sequence, tracking_age
        frame = np.asarray(row["target_12"], dtype=np.float64)
        if frame.shape != (12,) or not np.isfinite(frame).all():
            return None, "invalid_target", sequence, tracking_age
        if np.any(frame < 0.0) or np.any(frame > 1.0):
            return None, "target_out_of_range", sequence, tracking_age
        frame[0] = min(frame[0], LEFT_THUMB_NORMALIZED_MAX)
        return frame.astype(np.float32), "armed", sequence, tracking_age
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError):
        return None, "unavailable", None, None


def load_arm_offset(path, now, timeout, expected_mapping):
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        status, sequence, tracking_age = validate_common(
            row, now, timeout, ARM_SCHEMA, expected_mapping
        )
        if status != "armed":
            return None, status, sequence, tracking_age
        if row.get("reference") != "ready_wrist_yaw_pose":
            return None, "reference_rejected", sequence, tracking_age
        if row.get("frame") != "r1_root":
            return None, "frame_rejected", sequence, tracking_age
        if row.get("target_units") != "m":
            return None, "units_rejected", sequence, tracking_age
        if row.get("target_hand_order") != HAND_ORDER:
            return None, "hand_order_rejected", sequence, tracking_age
        offset = np.asarray(row["position_offset_m"], dtype=np.float64)
        if offset.shape != (6,) or not np.isfinite(offset).all():
            return None, "invalid_target", sequence, tracking_age
        if not is_dual_offset_in_workspace(offset):
            return None, "workspace_rejected", sequence, tracking_age
        return offset.astype(np.float32), "armed", sequence, tracking_age
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError):
        return None, "unavailable", None, None


def make_robot_cfg(urdf_path):
    return ArticulationCfg(
        prim_path="/World/R1A7O6",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            usd_dir="/tmp/r1_a7_o6_live_usd",
            fix_base=True,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=True,
            self_collision=True,
            activate_contact_sensors=True,
            make_instanceable=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=2,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=60.0,
                    damping=3.0,
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        actuators={
            "r1": ImplicitActuatorCfg(
                joint_names_expr=list(R1_JOINTS),
                effort_limit_sim=200.0,
                velocity_limit_sim=3.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "o6": ImplicitActuatorCfg(
                joint_names_expr=list(HAND_JOINTS),
                effort_limit_sim=10.0,
                velocity_limit_sim=3.0,
                stiffness=30.0,
                damping=1.0,
            ),
            "o6_mimic": ImplicitActuatorCfg(
                joint_names_expr=list(O6_MIMIC_JOINTS),
                effort_limit_sim=10.0,
                velocity_limit_sim=3.0,
                stiffness=0.0,
                damping=0.1,
            ),
        },
    )


def body_pose_in_root(robot, body_id):
    pose_w = robot.data.body_pose_w[:, body_id]
    root_w = robot.data.root_pose_w
    return subtract_frame_transforms(root_w[:, :3], root_w[:, 3:7], pose_w[:, :3], pose_w[:, 3:7])


def map_hand_target(target, joint_ids, normalized, limits):
    values = torch.as_tensor(normalized, device=target.device, dtype=target.dtype).unsqueeze(0)
    target[:, joint_ids] = limits[..., 0] + values * (limits[..., 1] - limits[..., 0])


def bounded_ik(controller, robot, body_id, jacobian_id, joint_ids, command, limits, posture):
    controller.set_command(command)
    ee_pos_b, ee_quat_b = body_pose_in_root(robot, body_id)
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_id, :, joint_ids]
    joint_pos = robot.data.joint_pos[:, joint_ids]
    _, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=True)
    if not torch.isfinite(singular_values).all():
        raise FloatingPointError("IK Jacobian has non-finite singular values")
    sigma_min = float(singular_values.min())
    condition = float(singular_values.max() / singular_values.min())
    if sigma_min < MIN_JACOBIAN_SINGULAR_VALUE or condition > MAX_JACOBIAN_CONDITION:
        raise FloatingPointError(
            f"IK Jacobian rejected: sigma_min={sigma_min:.6f} condition={condition:.1f} "
            f"posture_deviation={float(torch.max(torch.abs(joint_pos - posture))):.3f}"
        )
    desired = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
    null_basis = vh[:, singular_values.shape[-1] :, :]
    nullspace = torch.transpose(null_basis, 1, 2) @ null_basis
    desired += (
        NULLSPACE_POSTURE_GAIN
        * (nullspace @ (posture - joint_pos).unsqueeze(-1)).squeeze(-1)
    )
    if not torch.isfinite(desired).all():
        raise FloatingPointError("IK produced a non-finite joint target")
    if torch.any(desired < limits[..., 0]) or torch.any(desired > limits[..., 1]):
        raise FloatingPointError("IK candidate violated the 0.10 rad joint-limit margin")
    delta = desired - joint_pos
    max_delta = float(torch.max(torch.abs(delta)))
    if max_delta > MAX_ARM_JOINT_STEP_RAD:
        delta *= MAX_ARM_JOINT_STEP_RAD / max_delta
    return joint_pos + delta, sigma_min, condition


def rate_limit_offset(current, desired):
    result = current.copy()
    max_step = MAX_CARTESIAN_SPEED_M_S * PHYSICS_DT
    for base in (0, 3):
        delta = desired[base : base + 3] - current[base : base + 3]
        norm = float(np.linalg.norm(delta))
        if norm > max_step:
            delta *= max_step / norm
        result[base : base + 3] += delta
    return result


def main():
    if not (math.isfinite(args_cli.live_timeout) and args_cli.live_timeout > 0.0):
        raise ValueError("--live-timeout must be positive")
    if not (math.isfinite(args_cli.max_self_contact_force) and args_cli.max_self_contact_force > 0.0):
        raise ValueError("--max-self-contact-force must be positive")
    urdf_path = Path(args_cli.combined_urdf).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
    if not str(sim.device).startswith("cuda"):
        raise RuntimeError(f"R1-A7 O6 live preview requires CUDA, found {sim.device}")
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.82, 0.82, 0.82))
    light_cfg.func("/World/DomeLight", light_cfg)
    robot = Articulation(make_robot_cfg(urdf_path))
    contact_sensor = ContactSensor(
        ContactSensorCfg(prim_path="/World/R1A7O6/.*", update_period=0.0, debug_vis=False)
    )
    sim.set_camera_view(eye=[2.1, 1.8, 1.55], target=[0.25, 0.0, 1.0])
    sim.reset()
    robot.reset()

    left_arm_ids, left_arm_names = robot.find_joints(list(LEFT_ARM_JOINTS), preserve_order=True)
    right_arm_ids, right_arm_names = robot.find_joints(list(RIGHT_ARM_JOINTS), preserve_order=True)
    hand_ids, hand_names = robot.find_joints(list(HAND_JOINTS), preserve_order=True)
    if tuple(left_arm_names) != LEFT_ARM_JOINTS or tuple(right_arm_names) != RIGHT_ARM_JOINTS:
        raise RuntimeError(f"Unexpected R1 arm joint order: left={left_arm_names} right={right_arm_names}")
    if tuple(hand_names) != HAND_JOINTS:
        raise RuntimeError(f"Unexpected O6 live joint order: {hand_names}")
    left_body_ids, left_body_names = robot.find_bodies(["left_wrist_yaw_link"], preserve_order=True)
    right_body_ids, right_body_names = robot.find_bodies(["right_wrist_yaw_link"], preserve_order=True)
    if left_body_names != ["left_wrist_yaw_link"] or right_body_names != ["right_wrist_yaw_link"]:
        raise RuntimeError(f"Unexpected wrist bodies: {left_body_names} {right_body_names}")
    left_body_id = left_body_ids[0]
    right_body_id = right_body_ids[0]
    left_jacobian_id = left_body_id - 1 if robot.is_fixed_base else left_body_id
    right_jacobian_id = right_body_id - 1 if robot.is_fixed_base else right_body_id

    ready_target = robot.data.default_joint_pos.clone()
    name_to_id = {name: index for index, name in enumerate(robot.joint_names)}
    full_limits = robot.data.soft_joint_pos_limits
    for name, value in READY_ARM.items():
        joint_id = name_to_id[name]
        lower = float(full_limits[0, joint_id, 0])
        upper = float(full_limits[0, joint_id, 1])
        if not lower <= value <= upper:
            raise RuntimeError(f"Ready pose outside {name} limits")
        ready_target[:, joint_id] = value
    hand_limits = full_limits[:, hand_ids, :]
    map_hand_target(ready_target, hand_ids, np.zeros(12, dtype=np.float32), hand_limits)
    robot.write_joint_state_to_sim(ready_target, torch.zeros_like(ready_target))
    ready_contact_max = 0.0
    for _ in range(READY_SETTLE_STEPS):
        robot.set_joint_position_target(ready_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(PHYSICS_DT)
        contact_sensor.update(PHYSICS_DT, force_recompute=True)
        ready_contact_max = max(
            ready_contact_max,
            float(torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0], dim=-1).max()),
        )
    if ready_contact_max > args_cli.max_self_contact_force:
        raise RuntimeError(
            f"Ready pose self-contact {ready_contact_max:.3f} N exceeds {args_cli.max_self_contact_force:.3f} N"
        )
    root_quat = robot.data.root_quat_w
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=root_quat.device, dtype=root_quat.dtype)
    if not torch.allclose(root_quat, identity_quat, atol=1.0e-5):
        raise RuntimeError(f"Fixed root must remain world-aligned for Jacobian use: {root_quat}")

    left_ready_pos, left_ready_quat = body_pose_in_root(robot, left_body_id)
    right_ready_pos, right_ready_quat = body_pose_in_root(robot, right_body_id)
    left_ready_pos = left_ready_pos.clone()
    left_ready_quat = left_ready_quat.clone()
    right_ready_pos = right_ready_pos.clone()
    right_ready_quat = right_ready_quat.clone()
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": DLS_LAMBDA}
    )
    left_ik = DifferentialIKController(ik_cfg, num_envs=1, device=sim.device)
    right_ik = DifferentialIKController(ik_cfg, num_envs=1, device=sim.device)
    left_limits = full_limits[:, left_arm_ids, :].clone()
    right_limits = full_limits[:, right_arm_ids, :].clone()
    left_limits[..., 0] += JOINT_LIMIT_MARGIN_RAD
    left_limits[..., 1] -= JOINT_LIMIT_MARGIN_RAD
    right_limits[..., 0] += JOINT_LIMIT_MARGIN_RAD
    right_limits[..., 1] -= JOINT_LIMIT_MARGIN_RAD
    left_hold = robot.data.joint_pos[:, left_arm_ids].clone()
    right_hold = robot.data.joint_pos[:, right_arm_ids].clone()
    left_desired = left_hold.clone()
    right_desired = right_hold.clone()
    left_posture = left_hold.clone()
    right_posture = right_hold.clone()

    hand_path = Path(args_cli.hand_live_state)
    arm_path = Path(args_cli.arm_live_state)
    hand_status = None
    arm_status = None
    arm_tracking = False
    safety_latched = False
    safety_reason = None
    report_started = time.monotonic()
    sim_steps = 0
    hand_samples = 0
    arm_samples = 0
    last_hand_sequence = None
    last_arm_sequence = None
    invalid_hand_steps = 0
    invalid_arm_steps = 0
    max_contact_window = 0.0
    max_arm_speed_window = 0.0
    active_arm_offset = np.zeros(6, dtype=np.float32)
    accepted_arm_request = np.zeros(6, dtype=np.float32)
    sigma_min_window = math.inf
    condition_max_window = 0.0
    ik_settled = False
    physics_step_index = 0

    print("[R1-A7 O6 LIVE] fixed-base CAD mount; simulation only; DDS/network/serial are not used")
    print(f"[R1-A7 O6 LIVE] device={sim.device} hand_source={hand_path} arm_source={arm_path}")
    print(
        f"[R1-A7 O6 LIVE] ready_left_m={left_ready_pos[0].detach().cpu().tolist()} "
        f"ready_right_m={right_ready_pos[0].detach().cpu().tolist()}"
    )

    while simulation_app.is_running():
        now = time.monotonic()
        hand_frame, next_hand_status, hand_sequence, _ = load_hand_frame(
            hand_path, now, args_cli.live_timeout, args_cli.expected_hand_mapping
        )
        arm_offset, next_arm_status, arm_sequence, _ = load_arm_offset(
            arm_path, now, args_cli.live_timeout, args_cli.expected_arm_mapping
        )
        if next_hand_status != hand_status:
            print(
                f"[R1-A7 O6 LIVE] hand_state={next_hand_status} "
                f"output={'tracking' if hand_frame is not None else 'open'}"
            )
            hand_status = next_hand_status
        if next_arm_status != arm_status:
            print(
                f"[R1-A7 O6 LIVE] arm_state={next_arm_status} "
                f"output={'ik' if arm_offset is not None else 'hold'}"
            )
            arm_status = next_arm_status
        if hand_frame is not None and hand_sequence is not None and hand_sequence != last_hand_sequence:
            hand_samples += 1
            last_hand_sequence = hand_sequence
        if arm_offset is not None and arm_sequence is not None and arm_sequence != last_arm_sequence:
            arm_samples += 1
            last_arm_sequence = arm_sequence
        if hand_frame is None:
            hand_frame = np.zeros(12, dtype=np.float32)
            if hand_status not in ("unavailable", "stale", "disarmed"):
                invalid_hand_steps += 1
        if arm_offset is None and arm_status not in ("unavailable", "stale", "disarmed"):
            invalid_arm_steps += 1

        if next_arm_status == "workspace_rejected" and not safety_latched:
            safety_latched = True
            safety_reason = "workspace_rejected"
            print("[R1-A7 O6 LIVE] arm_state=workspace_rejected output=hold")

        if safety_latched:
            hand_frame = np.zeros(12, dtype=np.float32)
            arm_offset = None
        if arm_offset is not None:
            request_delta = np.asarray(arm_offset) - accepted_arm_request
            if max(
                np.linalg.norm(request_delta[:3]),
                np.linalg.norm(request_delta[3:]),
            ) >= ARM_COMMAND_WAKE_THRESHOLD_M:
                accepted_arm_request = np.asarray(arm_offset, dtype=np.float32).copy()
            arm_offset = accepted_arm_request
            next_active_arm_offset = rate_limit_offset(active_arm_offset, arm_offset)
            arm_command_changed = not np.allclose(
                next_active_arm_offset, active_arm_offset, atol=1.0e-9, rtol=0.0
            )
            active_arm_offset = next_active_arm_offset
            if arm_command_changed and ik_settled:
                print("[R1-A7 O6 LIVE] arm_control=moving output=ik")
                ik_settled = False
            left_command = torch.cat(
                (
                    left_ready_pos + torch.as_tensor(active_arm_offset[:3], device=sim.device).unsqueeze(0),
                    left_ready_quat,
                ),
                dim=1,
            )
            right_command = torch.cat(
                (
                    right_ready_pos + torch.as_tensor(active_arm_offset[3:], device=sim.device).unsqueeze(0),
                    right_ready_quat,
                ),
                dim=1,
            )
            left_goal_pos = left_command[:, :3]
            right_goal_pos = right_command[:, :3]
            wrist_distance = float(torch.linalg.vector_norm(left_goal_pos - right_goal_pos))
            if (
                wrist_distance < 0.20
                or float(left_goal_pos[0, 1]) <= 0.05
                or float(right_goal_pos[0, 1]) >= -0.05
            ):
                print(
                    f"[R1-A7 O6 LIVE] arm_state=workspace_rejected output=hold "
                    f"wrist_distance_m={wrist_distance:.3f}"
                )
                safety_latched = True
                safety_reason = "workspace_rejected"
                hand_frame = np.zeros(12, dtype=np.float32)
                arm_offset = None
        if arm_offset is not None:
            try:
                left_candidate, left_sigma_min, left_condition = bounded_ik(
                    left_ik,
                    robot,
                    left_body_id,
                    left_jacobian_id,
                    left_arm_ids,
                    left_command,
                    left_limits,
                    left_posture,
                )
                right_candidate, right_sigma_min, right_condition = bounded_ik(
                    right_ik,
                    robot,
                    right_body_id,
                    right_jacobian_id,
                    right_arm_ids,
                    right_command,
                    right_limits,
                    right_posture,
                )
                sigma_min_window = min(sigma_min_window, left_sigma_min, right_sigma_min)
                condition_max_window = max(condition_max_window, left_condition, right_condition)
                left_pos, _ = body_pose_in_root(robot, left_body_id)
                right_pos, _ = body_pose_in_root(robot, right_body_id)
                left_error = float(torch.linalg.vector_norm(left_goal_pos - left_pos))
                right_error = float(torch.linalg.vector_norm(right_goal_pos - right_pos))
                if (
                    not arm_command_changed
                    and left_error <= IK_SETTLE_POSITION_ERROR_M
                    and right_error <= IK_SETTLE_POSITION_ERROR_M
                ):
                    if not ik_settled:
                        print("[R1-A7 O6 LIVE] arm_control=settled output=joint_target_hold")
                    ik_settled = True
                if not ik_settled:
                    left_desired = left_candidate
                    right_desired = right_candidate
                arm_tracking = True
            except FloatingPointError as error:
                print(f"[R1-A7 O6 LIVE] arm_state=ik_fault output=hold error={error}")
                safety_latched = True
                safety_reason = "ik_fault"
                hand_frame = np.zeros(12, dtype=np.float32)
                arm_offset = None
        if arm_offset is None:
            if arm_tracking:
                left_hold = robot.data.joint_pos[:, left_arm_ids].clone()
                right_hold = robot.data.joint_pos[:, right_arm_ids].clone()
            left_desired = left_hold
            right_desired = right_hold
            arm_tracking = False
            ik_settled = False

        target = ready_target.clone()
        target[:, left_arm_ids] = left_desired
        target[:, right_arm_ids] = right_desired
        map_hand_target(target, hand_ids, hand_frame, hand_limits)
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=not args_cli.headless and physics_step_index % 2 == 0)
        physics_step_index += 1
        robot.update(PHYSICS_DT)
        contact_sensor.update(PHYSICS_DT, force_recompute=True)
        contact_force = float(torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0], dim=-1).max())
        max_contact_window = max(max_contact_window, contact_force)
        arm_speed = float(
            torch.max(
                torch.abs(
                    torch.cat(
                        (
                            robot.data.joint_vel[:, left_arm_ids],
                            robot.data.joint_vel[:, right_arm_ids],
                        ),
                        dim=1,
                    )
                )
            )
        )
        max_arm_speed_window = max(max_arm_speed_window, arm_speed)
        left_actual_pos, _ = body_pose_in_root(robot, left_body_id)
        right_actual_pos, _ = body_pose_in_root(robot, right_body_id)
        actual_offset = torch.cat(
            (left_actual_pos - left_ready_pos, right_actual_pos - right_ready_pos), dim=1
        )[0].detach().cpu().numpy()
        if (
            not is_dual_offset_in_workspace(actual_offset, ACTUAL_WORKSPACE_TOLERANCE_M)
            and not safety_latched
        ):
            safety_latched = True
            safety_reason = "actual_workspace_rejected"
            print("[R1-A7 O6 LIVE] arm_state=actual_workspace_rejected output=hold")
        if contact_force > args_cli.max_self_contact_force and not safety_latched:
            safety_latched = True
            safety_reason = "self_contact"
            print(
                f"[R1-A7 O6 LIVE] collision_hold force_n={contact_force:.3f} "
                f"threshold_n={args_cli.max_self_contact_force:.3f}"
            )
        if arm_speed > MAX_ARM_SPEED_RAD_S and not safety_latched:
            safety_latched = True
            safety_reason = "arm_overspeed"
            print(
                f"[R1-A7 O6 LIVE] arm_state=overspeed output=hold "
                f"speed_rad_s={arm_speed:.3f} threshold_rad_s={MAX_ARM_SPEED_RAD_S:.3f}"
            )

        sim_steps += 1
        report_now = time.monotonic()
        report_elapsed = report_now - report_started
        if report_elapsed >= 5.0:
            left_pos, _ = body_pose_in_root(robot, left_body_id)
            right_pos, _ = body_pose_in_root(robot, right_body_id)
            left_goal = left_ready_pos + torch.as_tensor(active_arm_offset[:3], device=sim.device).unsqueeze(0)
            right_goal = right_ready_pos + torch.as_tensor(active_arm_offset[3:], device=sim.device).unsqueeze(0)
            left_error_cm = float(torch.linalg.vector_norm(left_goal - left_pos) * 100.0)
            right_error_cm = float(torch.linalg.vector_norm(right_goal - right_pos) * 100.0)
            print(
                f"[R1-A7 O6 LIVE] sim_hz={sim_steps / report_elapsed:.1f} "
                f"hand_hz={hand_samples / report_elapsed:.1f} arm_hz={arm_samples / report_elapsed:.1f} "
                f"left_error_cm={left_error_cm:.2f} right_error_cm={right_error_cm:.2f} "
                f"max_contact_n={max_contact_window:.3f} max_arm_speed_rad_s={max_arm_speed_window:.3f} "
                f"sigma_min={sigma_min_window:.5f} condition_max={condition_max_window:.1f} "
                f"safety_latched={safety_latched} safety_reason={safety_reason} "
                f"invalid_hand_steps={invalid_hand_steps} invalid_arm_steps={invalid_arm_steps}"
            )
            report_started = report_now
            sim_steps = 0
            hand_samples = 0
            arm_samples = 0
            invalid_hand_steps = 0
            invalid_arm_steps = 0
            max_contact_window = 0.0
            max_arm_speed_window = 0.0
            sigma_min_window = math.inf
            condition_max_window = 0.0


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
