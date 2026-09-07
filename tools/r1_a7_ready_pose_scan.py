import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Scan simulation-only R1-A7 symmetric ready poses.")
parser.add_argument("--combined-urdf", default="/home/code/r1_a7_linker_o6.urdf")
parser.add_argument("--output", default="/home/code/reports/r1_a7_ready_pose_scan.json")
parser.add_argument("--verify-minimal", action="store_true")
parser.add_argument("--workspace-boundary", action="store_true")
parser.add_argument("--move-steps", type=int, default=360)
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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import subtract_frame_transforms


PHYSICS_DT = 1.0 / 60.0
READY_SETTLE_STEPS = 180
MAX_ARM_JOINT_STEP_RAD = 0.005
JOINT_LIMIT_MARGIN_RAD = 0.10
MIN_JACOBIAN_SINGULAR_VALUE = 0.005
MAX_JACOBIAN_CONDITION = 500.0
DESIGN_MIN_JACOBIAN_SINGULAR_VALUE = 0.010
DESIGN_MAX_JACOBIAN_CONDITION = 250.0
MAX_CARTESIAN_SPEED_M_S = 0.02
DLS_LAMBDA = 0.05
NULLSPACE_POSTURE_GAIN = 0.10
TARGET_X_M = 0.02
WORKSPACE_OFFSET_NORM_MAX_M = 0.06
WORKSPACE_SHOULDER_EXTENSION_MAX_M = 0.005

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
HAND_JOINTS = tuple(
    "lh_" + name
    for name in (
        "thumb_cmc_pitch",
        "thumb_cmc_yaw",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    )
) + tuple(
    "rh_" + name
    for name in (
        "thumb_cmc_pitch",
        "thumb_cmc_yaw",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    )
)
O6_MIMIC_JOINTS = tuple(
    "lh_" + name for name in ("thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip")
) + tuple(
    "rh_" + name for name in ("thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip")
)


def make_robot_cfg(urdf_path):
    return ArticulationCfg(
        prim_path="/World/R1A7O6Scan",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            usd_dir="/tmp/r1_a7_o6_ready_scan_usd",
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
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=60.0, damping=3.0)
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


def jacobian_metrics(robot, jacobian_id, joint_ids):
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_id, :, joint_ids]
    singular_values = torch.linalg.svdvals(jacobian)
    sigma_min = float(singular_values.min())
    condition = float(singular_values.max() / singular_values.min())
    return jacobian, sigma_min, condition


def bounded_ik(controller, robot, body_id, jacobian_id, joint_ids, command, limits, posture):
    controller.set_command(command)
    ee_pos_b, ee_quat_b = body_pose_in_root(robot, body_id)
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_id, :, joint_ids]
    joint_pos = robot.data.joint_pos[:, joint_ids]
    _, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=True)
    sigma_min = float(singular_values.min())
    condition = float(singular_values.max() / singular_values.min())
    if not math.isfinite(sigma_min) or not math.isfinite(condition):
        raise FloatingPointError("non-finite Jacobian metric")
    if sigma_min < MIN_JACOBIAN_SINGULAR_VALUE or condition > MAX_JACOBIAN_CONDITION:
        raise FloatingPointError(f"Jacobian gate sigma={sigma_min:.6f} condition={condition:.1f}")
    desired = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
    null_basis = vh[:, singular_values.shape[-1] :, :]
    nullspace = torch.transpose(null_basis, 1, 2) @ null_basis
    desired += NULLSPACE_POSTURE_GAIN * (nullspace @ (posture - joint_pos).unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(desired).all():
        raise FloatingPointError("non-finite IK target")
    if torch.any(desired < limits[..., 0]) or torch.any(desired > limits[..., 1]):
        raise FloatingPointError("joint-limit margin gate")
    delta = desired - joint_pos
    max_delta = float(torch.max(torch.abs(delta)))
    if max_delta > MAX_ARM_JOINT_STEP_RAD:
        delta *= MAX_ARM_JOINT_STEP_RAD / max_delta
    return joint_pos + delta, sigma_min, condition


def as_float(value):
    return round(float(value), 8)


def workspace_directions():
    directions = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                vector = np.asarray([x, y, z], dtype=np.float64)
                norm = float(np.linalg.norm(vector))
                if norm:
                    directions.append(vector / norm)
    return directions


def workspace_boundary_radius(ready_shoulder_vector, direction):
    ready_radius = float(np.linalg.norm(ready_shoulder_vector))
    projection = float(np.dot(ready_shoulder_vector, direction))
    extension_root = -projection + math.sqrt(
        projection**2
        + 2.0 * ready_radius * WORKSPACE_SHOULDER_EXTENSION_MAX_M
        + WORKSPACE_SHOULDER_EXTENSION_MAX_M**2
    )
    return min(WORKSPACE_OFFSET_NORM_MAX_M, extension_root)


def main():
    if not args_cli.headless:
        raise RuntimeError("Ready-pose scan must run headless")
    urdf_path = Path(args_cli.combined_urdf).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
    if not str(sim.device).startswith("cuda"):
        raise RuntimeError(f"Ready-pose scan requires CUDA, found {sim.device}")
    robot = Articulation(make_robot_cfg(urdf_path))
    contact_sensor = ContactSensor(
        ContactSensorCfg(prim_path="/World/R1A7O6Scan/.*", update_period=0.0, debug_vis=False)
    )
    sim.reset()
    robot.reset()

    left_ids, left_names = robot.find_joints(list(LEFT_ARM_JOINTS), preserve_order=True)
    right_ids, right_names = robot.find_joints(list(RIGHT_ARM_JOINTS), preserve_order=True)
    hand_ids, hand_names = robot.find_joints(list(HAND_JOINTS), preserve_order=True)
    if tuple(left_names) != LEFT_ARM_JOINTS or tuple(right_names) != RIGHT_ARM_JOINTS:
        raise RuntimeError("unexpected arm joint order")
    if tuple(hand_names) != HAND_JOINTS:
        raise RuntimeError("unexpected hand joint order")
    left_body_ids, _ = robot.find_bodies(["left_wrist_yaw_link"], preserve_order=True)
    right_body_ids, _ = robot.find_bodies(["right_wrist_yaw_link"], preserve_order=True)
    left_shoulder_body_ids, _ = robot.find_bodies(["left_shoulder_pitch_link"], preserve_order=True)
    right_shoulder_body_ids, _ = robot.find_bodies(["right_shoulder_pitch_link"], preserve_order=True)
    left_body_id = left_body_ids[0]
    right_body_id = right_body_ids[0]
    left_shoulder_body_id = left_shoulder_body_ids[0]
    right_shoulder_body_id = right_shoulder_body_ids[0]
    left_jacobian_id = left_body_id - 1
    right_jacobian_id = right_body_id - 1
    full_limits = robot.data.soft_joint_pos_limits
    left_limits = full_limits[:, left_ids, :].clone()
    right_limits = full_limits[:, right_ids, :].clone()
    left_limits[..., 0] += JOINT_LIMIT_MARGIN_RAD
    left_limits[..., 1] -= JOINT_LIMIT_MARGIN_RAD
    right_limits[..., 0] += JOINT_LIMIT_MARGIN_RAD
    right_limits[..., 1] -= JOINT_LIMIT_MARGIN_RAD
    hand_limits = full_limits[:, hand_ids, :]
    name_to_id = {name: index for index, name in enumerate(robot.joint_names)}

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": DLS_LAMBDA}
    )
    if args_cli.move_steps <= 0:
        raise ValueError("--move-steps must be positive")
    if args_cli.workspace_boundary:
        test_specs = [
            {
                "shoulder_pitch": -0.45,
                "elbow": 0.85,
                "wrist_pitch": 0.0,
                "direction": direction,
            }
            for direction in workspace_directions()
        ]
    else:
        candidates = (
            [(-0.45, 0.85, 0.00)]
            if args_cli.verify_minimal
            else [
                (shoulder_pitch, elbow, wrist_pitch)
                for shoulder_pitch in (-0.75, -0.60, -0.45)
                for elbow in (0.85, 1.00, 1.15)
                for wrist_pitch in (-0.25, 0.00, 0.25)
            ]
        )
        test_specs = [
            {
                "shoulder_pitch": shoulder_pitch,
                "elbow": elbow,
                "wrist_pitch": wrist_pitch,
                "direction": np.asarray([1.0, 0.0, 0.0]),
            }
            for shoulder_pitch, elbow, wrist_pitch in candidates
        ]
    results = []
    scan_started = time.monotonic()

    for candidate_index, spec in enumerate(test_specs, start=1):
        shoulder_pitch = spec["shoulder_pitch"]
        elbow = spec["elbow"]
        wrist_pitch = spec["wrist_pitch"]
        ready = robot.data.default_joint_pos.clone()
        ready_values = {
            "left_shoulder_pitch_joint": shoulder_pitch,
            "left_shoulder_roll_joint": 0.45,
            "left_elbow_joint": elbow,
            "left_wrist_pitch_joint": wrist_pitch,
            "right_shoulder_pitch_joint": shoulder_pitch,
            "right_shoulder_roll_joint": -0.45,
            "right_elbow_joint": elbow,
            "right_wrist_pitch_joint": wrist_pitch,
        }
        for name, value in ready_values.items():
            ready[:, name_to_id[name]] = value
        ready[:, hand_ids] = hand_limits[..., 0]
        robot.write_joint_state_to_sim(ready, torch.zeros_like(ready))
        ready_contact = 0.0
        for _ in range(READY_SETTLE_STEPS):
            robot.set_joint_position_target(ready)
            robot.write_data_to_sim()
            sim.step()
            robot.update(PHYSICS_DT)
            contact_sensor.update(PHYSICS_DT, force_recompute=True)
            ready_contact = max(
                ready_contact,
                float(torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0], dim=-1).max()),
            )

        _, left_ready_sigma, left_ready_condition = jacobian_metrics(robot, left_jacobian_id, left_ids)
        _, right_ready_sigma, right_ready_condition = jacobian_metrics(robot, right_jacobian_id, right_ids)
        left_ready_pos, left_ready_quat = body_pose_in_root(robot, left_body_id)
        right_ready_pos, right_ready_quat = body_pose_in_root(robot, right_body_id)
        left_ready_pos = left_ready_pos.clone()
        left_ready_quat = left_ready_quat.clone()
        right_ready_pos = right_ready_pos.clone()
        right_ready_quat = right_ready_quat.clone()
        if args_cli.workspace_boundary:
            left_shoulder_pos, _ = body_pose_in_root(robot, left_shoulder_body_id)
            right_shoulder_pos, _ = body_pose_in_root(robot, right_shoulder_body_id)
            left_direction = np.asarray(spec["direction"], dtype=np.float64)
            right_direction = left_direction * np.asarray([1.0, -1.0, 1.0])
            left_radius = workspace_boundary_radius(
                (left_ready_pos - left_shoulder_pos)[0].detach().cpu().numpy(), left_direction
            )
            right_radius = workspace_boundary_radius(
                (right_ready_pos - right_shoulder_pos)[0].detach().cpu().numpy(), right_direction
            )
            left_target_offset = left_radius * left_direction
            right_target_offset = right_radius * right_direction
        else:
            left_target_offset = np.asarray([TARGET_X_M, 0.0, 0.0])
            right_target_offset = left_target_offset.copy()
        path_steps = max(
            1,
            math.ceil(
                max(np.linalg.norm(left_target_offset), np.linalg.norm(right_target_offset))
                / (MAX_CARTESIAN_SPEED_M_S * PHYSICS_DT)
            ),
        )
        left_posture = robot.data.joint_pos[:, left_ids].clone()
        right_posture = robot.data.joint_pos[:, right_ids].clone()
        left_ik = DifferentialIKController(ik_cfg, num_envs=1, device=sim.device)
        right_ik = DifferentialIKController(ik_cfg, num_envs=1, device=sim.device)
        left_ik.reset()
        right_ik.reset()

        min_sigma = min(left_ready_sigma, right_ready_sigma)
        max_condition = max(left_ready_condition, right_ready_condition)
        move_contact = 0.0
        max_arm_speed = 0.0
        error_samples = []
        gate_error = None
        settled = False
        settled_hold_steps = 0
        left_desired = left_posture.clone()
        right_desired = right_posture.clone()
        executed_move_steps = 0
        for move_step in range(args_cli.move_steps):
            executed_move_steps = move_step + 1
            fraction = min(1.0, (move_step + 1) / path_steps)
            left_offset = torch.as_tensor(
                fraction * left_target_offset, device=sim.device, dtype=left_ready_pos.dtype
            ).unsqueeze(0)
            right_offset = torch.as_tensor(
                fraction * right_target_offset, device=sim.device, dtype=right_ready_pos.dtype
            ).unsqueeze(0)
            left_command = torch.cat((left_ready_pos + left_offset, left_ready_quat), dim=1)
            right_command = torch.cat((right_ready_pos + right_offset, right_ready_quat), dim=1)
            try:
                if settled:
                    _, left_sigma, left_condition = jacobian_metrics(robot, left_jacobian_id, left_ids)
                    _, right_sigma, right_condition = jacobian_metrics(robot, right_jacobian_id, right_ids)
                else:
                    left_desired, left_sigma, left_condition = bounded_ik(
                        left_ik,
                        robot,
                        left_body_id,
                        left_jacobian_id,
                        left_ids,
                        left_command,
                        left_limits,
                        left_posture,
                    )
                    right_desired, right_sigma, right_condition = bounded_ik(
                        right_ik,
                        robot,
                        right_body_id,
                        right_jacobian_id,
                        right_ids,
                        right_command,
                        right_limits,
                        right_posture,
                    )
            except FloatingPointError as error:
                gate_error = str(error)
                break
            min_sigma = min(min_sigma, left_sigma, right_sigma)
            max_condition = max(max_condition, left_condition, right_condition)
            target = ready.clone()
            target[:, left_ids] = left_desired
            target[:, right_ids] = right_desired
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(PHYSICS_DT)
            contact_sensor.update(PHYSICS_DT, force_recompute=True)
            move_contact = max(
                move_contact,
                float(torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0], dim=-1).max()),
            )
            max_arm_speed = max(
                max_arm_speed,
                float(torch.max(torch.abs(torch.cat((robot.data.joint_vel[:, left_ids], robot.data.joint_vel[:, right_ids]), dim=1)))),
            )
            left_pos, _ = body_pose_in_root(robot, left_body_id)
            right_pos, _ = body_pose_in_root(robot, right_body_id)
            left_error = float(torch.linalg.vector_norm(left_command[:, :3] - left_pos))
            right_error = float(torch.linalg.vector_norm(right_command[:, :3] - right_pos))
            error_samples.append(max(left_error, right_error))
            if (
                args_cli.workspace_boundary
                and move_step + 1 >= path_steps
                and left_error <= 0.001
                and right_error <= 0.001
            ):
                settled = True
            if settled:
                settled_hold_steps += 1
                if settled_hold_steps >= 60:
                    break

        left_pos, _ = body_pose_in_root(robot, left_body_id)
        right_pos, _ = body_pose_in_root(robot, right_body_id)
        left_goal = left_ready_pos + torch.as_tensor(
            left_target_offset, device=sim.device, dtype=left_ready_pos.dtype
        ).unsqueeze(0)
        right_goal = right_ready_pos + torch.as_tensor(
            right_target_offset, device=sim.device, dtype=right_ready_pos.dtype
        ).unsqueeze(0)
        final_left_error = float(torch.linalg.vector_norm(left_goal - left_pos))
        final_right_error = float(torch.linalg.vector_norm(right_goal - right_pos))
        tail = error_samples[-60:] if len(error_samples) >= 60 else error_samples
        tail_mean = float(np.mean(tail)) if tail else math.inf
        tail_span = float(max(tail) - min(tail)) if tail else math.inf
        final_arm_speed = float(
            torch.max(torch.abs(torch.cat((robot.data.joint_vel[:, left_ids], robot.data.joint_vel[:, right_ids]), dim=1)))
        )
        passed = (
            gate_error is None
            and min_sigma >= MIN_JACOBIAN_SINGULAR_VALUE
            and max_condition <= MAX_JACOBIAN_CONDITION
            and (
                not args_cli.workspace_boundary
                or (
                    settled_hold_steps >= 60
                    and
                    min_sigma >= DESIGN_MIN_JACOBIAN_SINGULAR_VALUE
                    and max_condition <= DESIGN_MAX_JACOBIAN_CONDITION
                )
            )
            and ready_contact <= 0.001
            and move_contact <= 0.001
            and max(final_left_error, final_right_error)
            <= (0.001 if args_cli.workspace_boundary else 0.003)
            and tail_span <= (0.001 if args_cli.workspace_boundary else 0.0005)
            and final_arm_speed <= 0.03
        )
        result = {
            "candidate": candidate_index,
            "shoulder_pitch_rad": shoulder_pitch,
            "elbow_rad": elbow,
            "wrist_pitch_rad": wrist_pitch,
            "direction": [as_float(value) for value in spec["direction"]],
            "left_target_offset_m": [as_float(value) for value in left_target_offset],
            "right_target_offset_m": [as_float(value) for value in right_target_offset],
            "executed_move_steps": executed_move_steps,
            "settled_hold_steps": settled_hold_steps,
            "ready_sigma_min": as_float(min(left_ready_sigma, right_ready_sigma)),
            "ready_condition_max": as_float(max(left_ready_condition, right_ready_condition)),
            "ready_contact_n": as_float(ready_contact),
            "move_min_sigma": as_float(min_sigma),
            "move_condition_max": as_float(max_condition),
            "move_contact_n": as_float(move_contact),
            "max_arm_speed_rad_s": as_float(max_arm_speed),
            "final_arm_speed_rad_s": as_float(final_arm_speed),
            "final_left_error_cm": as_float(final_left_error * 100.0),
            "final_right_error_cm": as_float(final_right_error * 100.0),
            "tail_error_mean_cm": as_float(tail_mean * 100.0),
            "tail_error_span_cm": as_float(tail_span * 100.0),
            "gate_error": gate_error,
            "pass": passed,
        }
        results.append(result)
        print("[READY SCAN] " + json.dumps(result, sort_keys=True), flush=True)

    passing = [row for row in results if row["pass"]]
    ranked = sorted(
        passing,
        key=lambda row: (
            -row["move_min_sigma"],
            row["final_left_error_cm"] + row["final_right_error_cm"],
            row["move_condition_max"],
        ),
    )
    dangerous_corner = None
    if args_cli.workspace_boundary:
        left_corner = np.asarray([0.12, 0.10, 0.0])
        right_corner = np.asarray([0.12, -0.10, 0.0])
        left_ready_vector = (left_ready_pos - left_shoulder_pos)[0].detach().cpu().numpy()
        right_ready_vector = (right_ready_pos - right_shoulder_pos)[0].detach().cpu().numpy()
        dangerous_corner = {
            "left_offset_m": left_corner.tolist(),
            "right_offset_m": right_corner.tolist(),
            "offset_norm_m": as_float(np.linalg.norm(left_corner)),
            "left_shoulder_extension_m": as_float(
                np.linalg.norm(left_ready_vector + left_corner) - np.linalg.norm(left_ready_vector)
            ),
            "right_shoulder_extension_m": as_float(
                np.linalg.norm(right_ready_vector + right_corner) - np.linalg.norm(right_ready_vector)
            ),
            "accepted_by_proposed_constraint": False,
            "executed": False,
        }
    report = {
        "schema": "r1_a7_workspace_cuda_validation_v1" if args_cli.workspace_boundary else "r1_a7_ready_pose_scan_v1",
        "simulation_only": True,
        "device": str(sim.device),
        "combined_urdf": str(urdf_path),
        "candidate_count": len(results),
        "pass_count": len(passing),
        "target": (
            "26 mirrored direction boundary points for both wrist_yaw_link"
            if args_cli.workspace_boundary
            else "both wrist_yaw_link +X 0.02 m in r1_root"
        ),
        "move_steps": args_cli.move_steps,
        "thresholds": {
            "sigma_min": MIN_JACOBIAN_SINGULAR_VALUE,
            "condition_max": MAX_JACOBIAN_CONDITION,
            "design_sigma_min": DESIGN_MIN_JACOBIAN_SINGULAR_VALUE,
            "design_condition_max": DESIGN_MAX_JACOBIAN_CONDITION,
            "contact_max_n": 0.001,
            "final_error_max_cm": 0.1 if args_cli.workspace_boundary else 0.3,
            "tail_error_span_max_cm": 0.1 if args_cli.workspace_boundary else 0.05,
            "final_arm_speed_max_rad_s": 0.03,
        },
        "recommended_candidate": ranked[0] if ranked else None,
        "proposed_workspace_constraint": (
            {
                "offset_norm_max_m": WORKSPACE_OFFSET_NORM_MAX_M,
                "shoulder_extension_max_m": WORKSPACE_SHOULDER_EXTENSION_MAX_M,
            }
            if args_cli.workspace_boundary
            else None
        ),
        "dangerous_corner_precheck": dangerous_corner,
        "results": results,
        "elapsed_s": as_float(time.monotonic() - scan_started),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[READY SCAN] report={output_path} pass={len(passing)}/{len(results)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
