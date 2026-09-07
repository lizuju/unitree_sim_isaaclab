import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build and smoke-test a STEP-mounted fixed-base R1-A7 + Linker O6 model.")
parser.add_argument(
    "--r1-urdf",
    default="/home/code/xr_teleoperate/assets/r1/r1_a7.urdf",
)
parser.add_argument(
    "--left-o6-urdf",
    default="/home/code/linkerhand-urdf/O6/left/linkerhand_o6_left.urdf",
)
parser.add_argument(
    "--right-o6-urdf",
    default="/home/code/linkerhand-urdf/O6/right/linkerhand_o6_right.urdf",
)
parser.add_argument(
    "--adapter-mesh",
    default="/home/code/r1-a7-o6/meshes/r1_a7_o6_adapter.stl",
)
parser.add_argument(
    "--left-bare-wrist-mesh",
    default="/home/code/r1-a7-o6/meshes/left_wrist_yaw_link_bare.stl",
)
parser.add_argument(
    "--right-bare-wrist-mesh",
    default="/home/code/r1-a7-o6/meshes/right_wrist_yaw_link_bare.stl",
)
parser.add_argument("--output-dir", default="/home/code/reports")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg


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
O6_DRIVER_SUFFIXES = (
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
O6_MIMIC_SUFFIXES = (
    "thumb_ip",
    "index_dip",
    "middle_dip",
    "ring_dip",
    "pinky_dip",
)
O6_ALL_SUFFIXES = (
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "thumb_ip",
    "index_mcp_pitch",
    "index_dip",
    "middle_mcp_pitch",
    "middle_dip",
    "ring_mcp_pitch",
    "ring_dip",
    "pinky_mcp_pitch",
    "pinky_dip",
)
O6_DRIVER_JOINTS = tuple("lh_" + name for name in O6_DRIVER_SUFFIXES) + tuple(
    "rh_" + name for name in O6_DRIVER_SUFFIXES
)
O6_MIMIC_JOINTS = tuple("lh_" + name for name in O6_MIMIC_SUFFIXES) + tuple(
    "rh_" + name for name in O6_MIMIC_SUFFIXES
)
O6_ALL_JOINTS = tuple("lh_" + name for name in O6_ALL_SUFFIXES) + tuple(
    "rh_" + name for name in O6_ALL_SUFFIXES
)
EXPECTED_DOF_NAMES = R1_JOINTS + O6_ALL_JOINTS
LOGICAL_DRIVER_JOINTS = R1_JOINTS + O6_DRIVER_JOINTS

MOUNTS = {
    "left": {
        "wrist_parent": "left_wrist_yaw_link",
        "adapter_link": "left_o6_adapter_link",
        "hand_child": "lh_hand_base_link",
        "wrist_to_adapter": {
            "xyz_m": (0.041, 0.0, -0.024),
            "rpy_rad": (math.pi / 2.0, 0.0, -math.pi / 2.0),
        },
        "adapter_to_hand": {
            "xyz_m": (0.0, 0.0, -0.013),
            "rpy_rad": (math.pi, 0.0, 0.0),
        },
    },
    "right": {
        "wrist_parent": "right_wrist_yaw_link",
        "adapter_link": "right_o6_adapter_link",
        "hand_child": "rh_hand_base_link",
        "wrist_to_adapter": {
            "xyz_m": (0.041, 0.0, 0.024),
            "rpy_rad": (math.pi / 2.0, 0.0, -math.pi / 2.0),
        },
        "adapter_to_hand": {
            "xyz_m": (0.0, 0.0, -0.013),
            "rpy_rad": (math.pi, 0.0, -math.pi),
        },
    },
}

ADAPTER_STEP_SHA256 = "03f5da5c0a501f8f2f97e3cec280c25ce4ed068d9ef5636011330afd13324d0a"
ADAPTER_INERTIAL = {
    "mass_kg": 0.05830189264537117,
    "origin_m": (-6.092385599176353e-06, 0.0006887867322979559, 0.0016257366024438795),
    "inertia_kg_m2": (
        1.3502534369572377e-05,
        -1.320690435279161e-09,
        7.412951001926512e-09,
        1.4262374716649116e-05,
        4.077433088814855e-07,
        1.2351130898010673e-05,
    ),
}
BARE_WRIST_INERTIALS = {
    "left": {
        "mass_kg": 0.07562417158720373,
        "origin_m": (0.017584058255833604, -0.00014749667968071403, -0.024491470236796604),
        "inertia_kg_m2": (
            6.0673379855505564e-05,
            -1.6880460283622178e-09,
            -6.497095165496116e-07,
            7.096689650597827e-05,
            -2.0310843017652346e-07,
            3.796021684661987e-05,
        ),
    },
    "right": {
        "mass_kg": 0.07562417497716631,
        "origin_m": (0.017321208146283077, -0.00015673081642568842, 0.024678010883387533),
        "inertia_kg_m2": (
            6.065689527110474e-05,
            -7.85265519671511e-09,
            8.92355869905765e-07,
            7.096687389785416e-05,
            2.051981909834244e-07,
            3.797673207134112e-05,
        ),
    },
}

PHYSICS_DT = 1.0 / 60.0
SETTLE_STEPS = 240
MAX_TRACKING_ERROR_RAD = 0.05
MIN_TIP_DISPLACEMENT_M = 0.03
MAX_SELF_CONTACT_FORCE_N = 1.0
MAX_MOUNT_DISTANCE_DRIFT_M = 1.0e-4


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_urdf(path, expected_root):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    links = [link.attrib["name"] for link in root.findall("link")]
    joints = [joint.attrib["name"] for joint in root.findall("joint")]
    child_links = {joint.find("child").attrib["link"] for joint in root.findall("joint")}
    root_links = sorted(set(links) - child_links)
    if root_links != [expected_root]:
        raise RuntimeError(f"Unexpected root link in {path}: {root_links}")
    if len(links) != len(set(links)) or len(joints) != len(set(joints)):
        raise RuntimeError(f"Duplicate link or joint names in {path}")
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if "://" in filename:
            continue
        mesh_path = (path.parent / filename).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh.set("filename", str(mesh_path))
    return path, root


def format_vector(values):
    return " ".join(f"{value:.10g}" for value in values)


def append_fixed_joint(root, name, parent, child, transform):
    joint = ET.SubElement(root, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": format_vector(transform["xyz_m"]),
            "rpy": format_vector(transform["rpy_rad"]),
        },
    )
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})


def set_inertial(link, values):
    inertial = link.find("inertial")
    inertial.find("origin").set("xyz", format_vector(values["origin_m"]))
    inertial.find("origin").set("rpy", "0 0 0")
    inertial.find("mass").set("value", f'{values["mass_kg"]:.12g}')
    ixx, ixy, ixz, iyy, iyz, izz = values["inertia_kg_m2"]
    inertial.find("inertia").attrib.update(
        {
            "ixx": f"{ixx:.12g}",
            "ixy": f"{ixy:.12g}",
            "ixz": f"{ixz:.12g}",
            "iyy": f"{iyy:.12g}",
            "iyz": f"{iyz:.12g}",
            "izz": f"{izz:.12g}",
        }
    )


def replace_stock_gripper(root, side, bare_wrist_mesh):
    link = root.find(f"./link[@name='{side}_wrist_yaw_link']")
    for mesh in link.findall(".//mesh"):
        mesh.set("filename", str(bare_wrist_mesh))
    set_inertial(link, BARE_WRIST_INERTIALS[side])


def append_adapter(root, side, mount, adapter_mesh):
    link = ET.SubElement(root, "link", {"name": mount["adapter_link"]})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": format_vector(ADAPTER_INERTIAL["origin_m"]), "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f'{ADAPTER_INERTIAL["mass_kg"]:.12g}'})
    ixx, ixy, ixz, iyy, iyz, izz = ADAPTER_INERTIAL["inertia_kg_m2"]
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{ixx:.12g}",
            "ixy": f"{ixy:.12g}",
            "ixz": f"{ixz:.12g}",
            "iyy": f"{iyy:.12g}",
            "iyz": f"{iyz:.12g}",
            "izz": f"{izz:.12g}",
        },
    )
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", {"filename": str(adapter_mesh)})
    ET.SubElement(visual, "material", {"name": "adapter_black"})
    # The positive-Z insert is enclosed by the wrist and must not be treated as an external contact surface.
    for name, z, radius, length in (("flange", -0.0065, 0.0215, 0.013),):
        collision = ET.SubElement(link, "collision", {"name": name})
        ET.SubElement(collision, "origin", {"xyz": f"0 0 {z:.10g}", "rpy": "0 0 0"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "cylinder", {"radius": str(radius), "length": str(length)})
    append_fixed_joint(
        root,
        f"{side}_o6_adapter_mount_joint",
        mount["wrist_parent"],
        mount["adapter_link"],
        mount["wrist_to_adapter"],
    )
    append_fixed_joint(
        root,
        f"{side}_o6_mount_joint",
        mount["adapter_link"],
        mount["hand_child"],
        mount["adapter_to_hand"],
    )


def build_combined_urdf(output_path):
    r1_path, r1_root = parse_urdf(args_cli.r1_urdf, "pelvis_link")
    left_path, left_root = parse_urdf(args_cli.left_o6_urdf, "lh_hand_base_link")
    right_path, right_root = parse_urdf(args_cli.right_o6_urdf, "rh_hand_base_link")
    adapter_mesh = Path(args_cli.adapter_mesh).expanduser().resolve(strict=True)
    bare_wrist_meshes = {
        "left": Path(args_cli.left_bare_wrist_mesh).expanduser().resolve(strict=True),
        "right": Path(args_cli.right_bare_wrist_mesh).expanduser().resolve(strict=True),
    }

    combined = deepcopy(r1_root)
    combined.set("name", "r1_a7_linker_o6")
    for side in ("left", "right"):
        replace_stock_gripper(combined, side, bare_wrist_meshes[side])
    material_names = {item.attrib["name"] for item in combined.findall("material")}
    for hand_root in (left_root, right_root):
        for material in hand_root.findall("material"):
            if material.attrib["name"] not in material_names:
                combined.append(deepcopy(material))
                material_names.add(material.attrib["name"])
        for element in hand_root:
            if element.tag in ("link", "joint"):
                combined.append(deepcopy(element))

    adapter_material = ET.Element("material", {"name": "adapter_black"})
    ET.SubElement(adapter_material, "color", {"rgba": "0.04 0.04 0.04 1"})
    combined.append(adapter_material)
    append_adapter(combined, "left", MOUNTS["left"], adapter_mesh)
    append_adapter(combined, "right", MOUNTS["right"], adapter_mesh)

    link_names = [link.attrib["name"] for link in combined.findall("link")]
    joint_names = [joint.attrib["name"] for joint in combined.findall("joint")]
    child_links = {joint.find("child").attrib["link"] for joint in combined.findall("joint")}
    root_links = sorted(set(link_names) - child_links)
    if len(link_names) != 44 or len(joint_names) != 43 or root_links != ["pelvis_link"]:
        raise RuntimeError(
            f"Unexpected combined URDF topology: links={len(link_names)} joints={len(joint_names)} roots={root_links}"
        )
    if len(link_names) != len(set(link_names)) or len(joint_names) != len(set(joint_names)):
        raise RuntimeError("Combined URDF contains duplicate link or joint names")

    ET.indent(combined, space="  ")
    ET.ElementTree(combined).write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "sources": {
            "r1_a7": {"path": str(r1_path), "sha256": file_sha256(r1_path)},
            "left_o6": {"path": str(left_path), "sha256": file_sha256(left_path)},
            "right_o6": {"path": str(right_path), "sha256": file_sha256(right_path)},
            "adapter_mesh": {"path": str(adapter_mesh), "sha256": file_sha256(adapter_mesh)},
            "left_bare_wrist_mesh": {
                "path": str(bare_wrist_meshes["left"]),
                "sha256": file_sha256(bare_wrist_meshes["left"]),
            },
            "right_bare_wrist_mesh": {
                "path": str(bare_wrist_meshes["right"]),
                "sha256": file_sha256(bare_wrist_meshes["right"]),
            },
        },
        "combined": {
            "path": str(output_path),
            "sha256": file_sha256(output_path),
            "root_link": "pelvis_link",
            "link_count": len(link_names),
            "source_joint_count": len(joint_names),
            "fixed_mount_joint_count": 4,
            "expected_articulation_dof_count": len(EXPECTED_DOF_NAMES),
            "logical_driver_joint_count": len(LOGICAL_DRIVER_JOINTS),
            "o6_mimic_joint_count": len(O6_MIMIC_JOINTS),
        },
    }


def make_robot_cfg(combined_urdf):
    return ArticulationCfg(
        prim_path="/World/R1A7O6",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(combined_urdf),
            usd_dir="/tmp/r1_a7_o6_usd",
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
                joint_names_expr=list(O6_DRIVER_JOINTS),
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


def collect_body_positions(robot, body_names):
    body_ids, resolved = robot.find_bodies(list(body_names), preserve_order=True)
    if tuple(resolved) != tuple(body_names):
        raise RuntimeError(f"Unexpected body order: {resolved}")
    values = robot.data.body_pos_w[0, body_ids].detach().cpu()
    return {name: [float(value) for value in values[index]] for index, name in enumerate(body_names)}


def vector_distance(left, right):
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def make_reach_target(robot, joint_ids):
    target = torch.zeros_like(robot.data.default_joint_pos)
    values = {
        "left_shoulder_pitch_joint": -0.60,
        "left_shoulder_roll_joint": 0.45,
        "left_elbow_joint": 1.00,
        "right_shoulder_pitch_joint": -0.60,
        "right_shoulder_roll_joint": -0.45,
        "right_elbow_joint": 1.00,
    }
    index_by_name = {name: joint_ids[index] for index, name in enumerate(EXPECTED_DOF_NAMES)}
    limits = robot.data.soft_joint_pos_limits[0]
    for name, value in values.items():
        joint_id = index_by_name[name]
        lower = float(limits[joint_id, 0])
        upper = float(limits[joint_id, 1])
        if not lower <= value <= upper:
            raise RuntimeError(f"Reach target outside {name} limits: {value} not in [{lower}, {upper}]")
        target[:, joint_id] = value
    return target


def settle(robot, contact_sensor, target, steps):
    max_contact_force = 0.0
    max_contact_body = None
    for _ in range(steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(PHYSICS_DT)
        contact_sensor.update(PHYSICS_DT, force_recompute=True)
        forces = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0], dim=-1)
        force, body_id = torch.max(forces, dim=0)
        if float(force) > max_contact_force:
            max_contact_force = float(force)
            max_contact_body = contact_sensor.body_names[int(body_id)]
    return max_contact_force, max_contact_body


def main():
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_urdf = output_dir / "r1_a7_linker_o6.urdf"
    asset_report = build_combined_urdf(combined_urdf)

    global sim
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
    if not str(sim.device).startswith("cuda"):
        raise RuntimeError(f"This smoke test requires CUDA, found device={sim.device}")
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/DomeLight", light_cfg)
    robot = Articulation(make_robot_cfg(combined_urdf))
    contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/R1A7O6/.*",
            update_period=0.0,
            debug_vis=False,
        )
    )
    sim.set_camera_view(eye=[1.8, 1.8, 1.5], target=[0.0, 0.0, 1.0])
    sim.reset()
    robot.reset()

    joint_ids, joint_names = robot.find_joints(list(EXPECTED_DOF_NAMES), preserve_order=True)
    if tuple(joint_names) != EXPECTED_DOF_NAMES:
        raise RuntimeError(f"Unexpected articulation joint order: {joint_names}")
    if robot.num_joints != len(EXPECTED_DOF_NAMES):
        raise RuntimeError(
            f"Expected {len(EXPECTED_DOF_NAMES)} articulation DOFs including tendon-coupled O6 mimic joints, "
            f"found {robot.num_joints}: {robot.joint_names}"
        )

    observed_bodies = (
        "left_wrist_yaw_link",
        "left_o6_adapter_link",
        "lh_hand_base_link",
        "lh_index_distal",
        "right_wrist_yaw_link",
        "right_o6_adapter_link",
        "rh_hand_base_link",
        "rh_index_distal",
    )
    zero_target = torch.zeros_like(robot.data.default_joint_pos)
    robot.write_joint_state_to_sim(zero_target, torch.zeros_like(zero_target))
    neutral_force, neutral_force_body = settle(robot, contact_sensor, zero_target, SETTLE_STEPS)
    neutral_positions = collect_body_positions(robot, observed_bodies)

    reach_target = make_reach_target(robot, joint_ids)
    reach_force, reach_force_body = settle(robot, contact_sensor, reach_target, SETTLE_STEPS)
    reach_positions = collect_body_positions(robot, observed_bodies)
    tracking_error = torch.abs(robot.data.joint_pos - reach_target)
    max_tracking_error = float(torch.max(tracking_error))
    tracking_error_values = tracking_error[0].detach().cpu().tolist()
    joint_tracking_error = {
        name: float(tracking_error_values[joint_id])
        for name, joint_id in zip(EXPECTED_DOF_NAMES, joint_ids)
    }
    worst_tracking_joint = max(joint_tracking_error, key=joint_tracking_error.get)

    left_tip_displacement = vector_distance(
        neutral_positions["lh_index_distal"], reach_positions["lh_index_distal"]
    )
    right_tip_displacement = vector_distance(
        neutral_positions["rh_index_distal"], reach_positions["rh_index_distal"]
    )
    mount_distance_drift = {}
    for side, wrist, adapter, hand in (
        ("left", "left_wrist_yaw_link", "left_o6_adapter_link", "lh_hand_base_link"),
        ("right", "right_wrist_yaw_link", "right_o6_adapter_link", "rh_hand_base_link"),
    ):
        for segment, parent, child in (
            ("wrist_adapter", wrist, adapter),
            ("adapter_hand", adapter, hand),
        ):
            neutral_distance = vector_distance(neutral_positions[parent], neutral_positions[child])
            reach_distance = vector_distance(reach_positions[parent], reach_positions[child])
            mount_distance_drift[f"{side}_{segment}"] = abs(reach_distance - neutral_distance)

    max_contact_force = max(neutral_force, reach_force)
    checks = {
        "joint_count": robot.num_joints == len(EXPECTED_DOF_NAMES),
        "finite_joint_state": bool(torch.isfinite(robot.data.joint_pos).all()),
        "reach_tracking": max_tracking_error <= MAX_TRACKING_ERROR_RAD,
        "left_tip_moves": left_tip_displacement >= MIN_TIP_DISPLACEMENT_M,
        "right_tip_moves": right_tip_displacement >= MIN_TIP_DISPLACEMENT_M,
        "fixed_mounts": max(mount_distance_drift.values()) <= MAX_MOUNT_DISTANCE_DRIFT_M,
        "no_self_collision": max_contact_force <= MAX_SELF_CONTACT_FORCE_N,
    }
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"r1_a7_o6_combined_smoke_{run_id}.json"
    report = {
        "schema": "r1_a7_linker_o6_combined_smoke_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(checks.values()),
        "scope": "fixed-base Isaac simulation smoke only; no DDS, serial, or physical-hand command path",
        "mount_status": "CAD_HOLE_REGISTERED_USER_PHYSICAL_ORIENTATION_PENDING_PHYSICAL_MEASUREMENT",
        "mount_basis": (
            "STEP shoulder at the bare R1 wrist; O6 20x20 mm four-hole pattern and 2 mm locating bosses; "
            "viewed from the robot center toward the fingertips, left O6 is twisted 90 deg counterclockwise "
            "and right O6 90 deg clockwise from the previous photo-derived phase"
        ),
        "mounts": MOUNTS,
        "adapter_geometry": {
            "source_step_sha256": ADAPTER_STEP_SHA256,
            "normalized_frame": "origin at R1 shoulder face; +Z points into R1; -Z points to O6",
            "bbox_m": [[-0.0215, -0.0215, -0.015], [0.0215, 0.0215, 0.02689479]],
            "o6_hole_centers_m": [[-0.01, -0.01], [-0.01, 0.01], [0.01, -0.01], [0.01, 0.01]],
            "o6_main_face_m": -0.015,
            "o6_root_origin_m": -0.013,
            "collision_model": "external flange only; the internal wrist insert is intentionally excluded",
        },
        "inertial_basis": {
            "bare_wrists": "volume-proportional split calibrated to the original 0.63414044 kg wrist-link mass",
            "adapter": "uniform 2700 kg/m3 assumption because the STEP file contains no material density",
            "status": "PROVISIONAL_PENDING_PHYSICAL_MASS_MEASUREMENT",
        },
        "assets": asset_report,
        "simulation": {
            "device": str(sim.device),
            "physics_dt_s": PHYSICS_DT,
            "fixed_base": True,
            "self_collisions_enabled": True,
            "merge_fixed_joints": False,
            "convert_mimic_joints_to_normal_joints": True,
            "o6_mimic_implementation": "Isaac Sim 5.1 parse_mimic using PhysX tendons",
            "articulation_dof_count": robot.num_joints,
            "logical_driver_joint_count": len(LOGICAL_DRIVER_JOINTS),
            "articulation_body_count": robot.num_bodies,
            "joint_names": list(robot.joint_names),
        },
        "smoke": {
            "settle_steps_per_pose": SETTLE_STEPS,
            "neutral_body_positions_m": neutral_positions,
            "reach_body_positions_m": reach_positions,
            "left_tip_displacement_m": left_tip_displacement,
            "right_tip_displacement_m": right_tip_displacement,
            "max_joint_tracking_error_rad": max_tracking_error,
            "worst_tracking_joint": worst_tracking_joint,
            "joint_tracking_error_rad": joint_tracking_error,
            "mount_distance_drift_m": mount_distance_drift,
            "max_self_contact_force_n": max_contact_force,
            "max_contact_body": neutral_force_body if neutral_force >= reach_force else reach_force_body,
            "neutral_max_self_contact_force_n": neutral_force,
            "neutral_max_contact_body": neutral_force_body,
            "reach_max_self_contact_force_n": reach_force,
            "reach_max_contact_body": reach_force_body,
        },
        "thresholds": {
            "max_joint_tracking_error_rad": MAX_TRACKING_ERROR_RAD,
            "min_tip_displacement_m": MIN_TIP_DISPLACEMENT_M,
            "max_mount_distance_drift_m": MAX_MOUNT_DISTANCE_DRIFT_M,
            "max_self_contact_force_n": MAX_SELF_CONTACT_FORCE_N,
        },
        "checks": checks,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f'[R1-A7 O6] result={"PASS" if report["overall_pass"] else "FAIL"}')
    print(f"[R1-A7 O6] combined_urdf={combined_urdf}")
    print(f"[R1-A7 O6] report={report_path}")
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
