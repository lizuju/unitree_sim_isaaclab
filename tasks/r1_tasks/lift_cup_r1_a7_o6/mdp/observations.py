from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pose_in_robot_root(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    cup: RigidObject = env.scene[object_cfg.name]
    pos, quat = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, cup.data.root_pos_w, cup.data.root_quat_w
    )
    return torch.cat((pos, quat), dim=-1)


def object_velocity_in_robot_root(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    cup: RigidObject = env.scene[object_cfg.name]
    linear = quat_apply_inverse(robot.data.root_quat_w, cup.data.root_lin_vel_w)
    angular = quat_apply_inverse(robot.data.root_quat_w, cup.data.root_ang_vel_w)
    return torch.cat((linear, angular), dim=-1)


def body_positions_in_robot_root(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, body_cfg.body_ids]
    count = body_pos_w.shape[1]
    root_pos_w = robot.data.root_pos_w[:, None, :].expand(-1, count, -1)
    root_quat_w = robot.data.root_quat_w[:, None, :].expand(-1, count, -1)
    body_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w)
    return body_pos_b.reshape(env.num_envs, -1)


def grasp_point_to_object(
    env: ManagerBasedRLEnv,
    palm_cfg: SceneEntityCfg,
    offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    cup: RigidObject = env.scene[object_cfg.name]
    palm_pos = robot.data.body_pos_w[:, palm_cfg.body_ids[0]]
    palm_quat = robot.data.body_quat_w[:, palm_cfg.body_ids[0]]
    local_offset = torch.tensor(offset, device=env.device).expand(env.num_envs, -1)
    grasp_pos = palm_pos + quat_apply(palm_quat, local_offset)
    delta_w = cup.data.root_pos_w - grasp_pos
    return quat_apply_inverse(robot.data.root_quat_w, delta_w)


def contact_forces_in_robot_root(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)
    root_quat_w = robot.data.root_quat_w[:, None, :].expand(-1, force_w.shape[1], -1)
    return quat_apply_inverse(root_quat_w, force_w).reshape(env.num_envs, -1)
