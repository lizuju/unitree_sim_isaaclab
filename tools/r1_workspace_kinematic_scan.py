#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


DT = 1.0 / 60.0
CARTESIAN_SPEED_M_S = 0.02
MAX_JOINT_STEP_RAD = 0.005
DLS_LAMBDA = 0.05
NULLSPACE_POSTURE_GAIN = 0.10
JOINT_LIMIT_MARGIN_RAD = 0.10
RUNTIME_SIGMA_MIN = 0.005
RUNTIME_CONDITION_MAX = 500.0
DESIGN_SIGMA_MIN = 0.010
DESIGN_CONDITION_MAX = 250.0
SETTLE_POSITION_ERROR_M = 0.001
PROPOSED_OFFSET_NORM_MAX_M = 0.06
PROPOSED_SHOULDER_EXTENSION_MAX_M = 0.005

READY = {
    "shoulder_pitch_joint": -0.45,
    "shoulder_roll_joint": 0.45,
    "shoulder_yaw_joint": 0.0,
    "elbow_joint": 0.85,
    "wrist_roll_joint": 0.0,
    "wrist_pitch_joint": 0.0,
    "wrist_yaw_joint": 0.0,
}


def rotation_axis(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ]
    )


def rotation_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def rotation_error(target, current):
    error_rotation = target @ current.T
    cosine = np.clip((np.trace(error_rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cosine)
    skew = np.asarray(
        [
            error_rotation[2, 1] - error_rotation[1, 2],
            error_rotation[0, 2] - error_rotation[2, 0],
            error_rotation[1, 0] - error_rotation[0, 1],
        ]
    )
    if angle < 1.0e-8:
        return 0.5 * skew
    return angle * skew / (2.0 * math.sin(angle))


def parse_arm_chain(urdf_path, side):
    root = ET.parse(urdf_path).getroot()
    child_to_joint = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}
    chain = []
    link = f"{side}_wrist_yaw_link"
    while link in child_to_joint:
        joint = child_to_joint[link]
        chain.append(joint)
        link = joint.find("parent").attrib["link"]
    chain.reverse()
    result = []
    for joint in chain:
        if joint.attrib["type"] not in ("revolute", "continuous"):
            continue
        name = joint.attrib["name"]
        if not name.startswith(f"{side}_"):
            continue
        origin = joint.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
        limit = joint.find("limit")
        result.append(
            {
                "name": name,
                "short_name": name.removeprefix(f"{side}_"),
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
                "lower": float(limit.attrib["lower"]) + JOINT_LIMIT_MARGIN_RAD,
                "upper": float(limit.attrib["upper"]) - JOINT_LIMIT_MARGIN_RAD,
            }
        )
    return result


def forward(chain, joint_position):
    transform = np.eye(4)
    joint_records = []
    for joint, angle in zip(chain, joint_position, strict=True):
        origin = np.eye(4)
        origin[:3, :3] = rotation_rpy(joint["rpy"])
        origin[:3, 3] = joint["xyz"]
        transform = transform @ origin
        axis_world = transform[:3, :3] @ joint["axis"]
        joint_records.append((transform[:3, 3].copy(), axis_world.copy()))
        rotation = np.eye(4)
        rotation[:3, :3] = rotation_axis(joint["axis"], angle)
        transform = transform @ rotation
    position = transform[:3, 3]
    jacobian_columns = [
        np.concatenate((np.cross(axis, position - joint_position_world), axis))
        for joint_position_world, axis in joint_records
    ]
    return position, transform[:3, :3], np.stack(jacobian_columns, axis=1)


def metrics(jacobian):
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return float(singular_values[-1]), float(singular_values[0] / singular_values[-1])


def evaluate_offset(chain, ready_joint_position, offset):
    ready_position, ready_rotation, _ = forward(chain, ready_joint_position)
    target_position = ready_position + offset
    joint_position = ready_joint_position.copy()
    min_sigma = math.inf
    max_condition = 0.0
    gate = None
    path_steps = max(1, math.ceil(np.linalg.norm(offset) / (CARTESIAN_SPEED_M_S * DT)))
    last_position_error = math.inf
    for step in range(path_steps + 360):
        fraction = min(1.0, (step + 1) / path_steps)
        command_position = ready_position + fraction * offset
        position, rotation, jacobian = forward(chain, joint_position)
        sigma_min, condition = metrics(jacobian)
        min_sigma = min(min_sigma, sigma_min)
        max_condition = max(max_condition, condition)
        if sigma_min < RUNTIME_SIGMA_MIN or condition > RUNTIME_CONDITION_MAX:
            gate = "runtime_jacobian"
            break
        error = np.concatenate((command_position - position, rotation_error(ready_rotation, rotation)))
        last_position_error = float(np.linalg.norm(error[:3]))
        if step >= path_steps and last_position_error <= SETTLE_POSITION_ERROR_M:
            break
        task_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + DLS_LAMBDA**2 * np.eye(6), error
        )
        _, _, vh = np.linalg.svd(jacobian, full_matrices=True)
        null_basis = vh[6:, :]
        nullspace = null_basis.T @ null_basis
        joint_delta = task_delta + NULLSPACE_POSTURE_GAIN * nullspace @ (
            ready_joint_position - joint_position
        )
        max_delta = float(np.max(np.abs(joint_delta)))
        if max_delta > MAX_JOINT_STEP_RAD:
            joint_delta *= MAX_JOINT_STEP_RAD / max_delta
        candidate = joint_position + joint_delta
        lower = np.asarray([joint["lower"] for joint in chain])
        upper = np.asarray([joint["upper"] for joint in chain])
        if np.any(candidate < lower) or np.any(candidate > upper):
            gate = "joint_limit_margin"
            break
        joint_position = candidate
    final_position, _, final_jacobian = forward(chain, joint_position)
    final_sigma, final_condition = metrics(final_jacobian)
    final_error = float(np.linalg.norm(target_position - final_position))
    runtime_pass = (
        gate is None
        and min_sigma >= RUNTIME_SIGMA_MIN
        and max_condition <= RUNTIME_CONDITION_MAX
        and final_error <= SETTLE_POSITION_ERROR_M
    )
    design_pass = (
        runtime_pass
        and min_sigma >= DESIGN_SIGMA_MIN
        and max_condition <= DESIGN_CONDITION_MAX
    )
    return {
        "runtime_pass": runtime_pass,
        "design_pass": design_pass,
        "gate": gate,
        "min_sigma": round(min_sigma, 8),
        "max_condition": round(max_condition, 4),
        "final_sigma": round(final_sigma, 8),
        "final_condition": round(final_condition, 4),
        "final_error_m": round(final_error, 8),
    }


def unit_directions():
    directions = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                vector = np.asarray([x, y, z])
                norm = np.linalg.norm(vector)
                if norm:
                    directions.append(vector / norm)
    return directions


def fibonacci_directions(count):
    directions = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        azimuth = golden_angle * index
        directions.append(np.asarray([radial * math.cos(azimuth), radial * math.sin(azimuth), z]))
    return directions


def proposed_boundary_radius(ready_shoulder_vector, direction):
    ready_radius = float(np.linalg.norm(ready_shoulder_vector))
    projection = float(np.dot(ready_shoulder_vector, direction))
    extension_root = -projection + math.sqrt(
        projection**2
        + 2.0 * ready_radius * PROPOSED_SHOULDER_EXTENSION_MAX_M
        + PROPOSED_SHOULDER_EXTENSION_MAX_M**2
    )
    return min(PROPOSED_OFFSET_NORM_MAX_M, extension_root)


def main():
    parser = argparse.ArgumentParser(description="Offline fixed-orientation R1-A7 wrist workspace scan.")
    parser.add_argument("--urdf", default=".tmp_xr_o6_stage1/reports/r1_a7_linker_o6.urdf")
    parser.add_argument("--output", default=".tmp_xr_o6_stage1/reports/r1_workspace_kinematic_scan.json")
    args = parser.parse_args()
    urdf_path = Path(args.urdf).resolve()
    rows = []
    boundary_rows = []
    ready_geometry = {}
    for side in ("left", "right"):
        chain = parse_arm_chain(urdf_path, side)
        ready_joint_position = np.asarray(
            [
                (-READY["shoulder_roll_joint"] if side == "right" and joint["short_name"] == "shoulder_roll_joint" else READY[joint["short_name"]])
                for joint in chain
            ]
        )
        ready_position, _, ready_jacobian = forward(chain, ready_joint_position)
        ready_sigma, ready_condition = metrics(ready_jacobian)
        shoulder_position = chain[0]["xyz"]
        ready_shoulder_vector = ready_position - shoulder_position
        ready_geometry[side] = {
            "wrist_position_m": [round(float(value), 8) for value in ready_position],
            "shoulder_position_m": [round(float(value), 8) for value in shoulder_position],
            "shoulder_to_wrist_radius_m": round(float(np.linalg.norm(ready_shoulder_vector)), 8),
            "sigma_min": round(ready_sigma, 8),
            "condition": round(ready_condition, 4),
        }
        for direction_index, direction in enumerate(unit_directions(), start=1):
            for distance_m in np.arange(0.01, 0.121, 0.01):
                offset = direction * distance_m
                result = evaluate_offset(chain, ready_joint_position, offset)
                rows.append(
                    {
                        "side": side,
                        "direction_index": direction_index,
                        "direction": [round(float(value), 8) for value in direction],
                        "distance_m": round(float(distance_m), 3),
                        "offset_m": [round(float(value), 8) for value in offset],
                        **result,
                    }
                )
        for direction_index, direction in enumerate(fibonacci_directions(256), start=1):
            radius = proposed_boundary_radius(ready_shoulder_vector, direction)
            offset = radius * direction
            result = evaluate_offset(chain, ready_joint_position, offset)
            boundary_rows.append(
                {
                    "side": side,
                    "direction_index": direction_index,
                    "direction": [round(float(value), 8) for value in direction],
                    "radius_m": round(float(radius), 8),
                    "offset_m": [round(float(value), 8) for value in offset],
                    **result,
                }
            )
    report = {
        "schema": "r1_workspace_kinematic_scan_v1",
        "method": "URDF geometric Jacobian plus DLS and exact nullspace posture; no dynamics or collision model",
        "ready": READY,
        "runtime_thresholds": {"sigma_min": RUNTIME_SIGMA_MIN, "condition_max": RUNTIME_CONDITION_MAX},
        "design_thresholds": {"sigma_min": DESIGN_SIGMA_MIN, "condition_max": DESIGN_CONDITION_MAX},
        "ready_geometry": ready_geometry,
        "proposed_constraint": {
            "offset_norm_max_m": PROPOSED_OFFSET_NORM_MAX_M,
            "shoulder_extension_max_m": PROPOSED_SHOULDER_EXTENSION_MAX_M,
            "formula": "norm(offset)<=0.06 and norm((ready_wrist-shoulder)+offset)<=norm(ready_wrist-shoulder)+0.005",
            "dense_boundary_direction_count_per_arm": 256,
        },
        "proposed_boundary_rows": boundary_rows,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
