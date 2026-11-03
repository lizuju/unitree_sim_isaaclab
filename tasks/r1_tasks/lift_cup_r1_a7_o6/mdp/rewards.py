from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _fingertip_points_w(
    robot: Articulation,
    body_ids: list[int],
    offsets: list[tuple[float, float, float]],
) -> torch.Tensor:
    body_pos = robot.data.body_pos_w[:, body_ids]
    body_quat = robot.data.body_quat_w[:, body_ids]
    local_offsets = torch.tensor(offsets, device=robot.device).expand(body_pos.shape[0], -1, -1)
    return body_pos + quat_apply(body_quat, local_offsets)


def qualified_contact(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    threshold: float,
    fingertips_cfg: SceneEntityCfg,
    fingertip_offsets: list[tuple[float, float, float]],
    nominal_xy: tuple[float, float],
    drift_tolerance: float,
    drift_scale: float,
    min_upright: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    robot: Articulation = env.scene[fingertips_cfg.name]
    cup: RigidObject = env.scene[object_cfg.name]
    magnitudes = [
        torch.linalg.vector_norm(env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3), dim=-1)
        for name in contact_sensor_names
    ]
    contacts = torch.stack(magnitudes, dim=1) > threshold

    tip_pos = _fingertip_points_w(robot, fingertips_cfg.body_ids, fingertip_offsets)
    cup_quat = cup.data.root_quat_w[:, None, :].expand(-1, tip_pos.shape[1], -1)
    tip_pos_cup = quat_apply_inverse(cup_quat, tip_pos - cup.data.root_pos_w[:, None, :])
    radial = tip_pos_cup[..., :2]
    radial_direction = radial / torch.linalg.vector_norm(radial, dim=-1).clamp_min(1.0e-6).unsqueeze(-1)
    opposition = (1.0 - torch.sum(radial_direction[:, :1] * radial_direction[:, 1:], dim=-1)) * 0.5
    opposition_gate = ((opposition - 0.5) / 0.5).clamp(0.0, 1.0)
    pair_quality = torch.max((contacts[:, :1] & contacts[:, 1:]).float() * opposition_gate, dim=1).values

    local_z = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, -1)
    up_z = quat_apply(cup.data.root_quat_w, local_z)[:, 2]
    upright_gate = ((up_z - min_upright) / (1.0 - min_upright)).clamp(0.0, 1.0)

    cup_xy = cup.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    target_xy = torch.tensor(nominal_xy, device=env.device).expand(env.num_envs, -1)
    drift = torch.linalg.vector_norm(cup_xy - target_xy, dim=-1)
    drift_gate = 1.0 - ((drift - drift_tolerance) / drift_scale).clamp(0.0, 1.0)
    return pair_quality * upright_gate * drift_gate


def fingertip_contact_count(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    threshold: float,
) -> torch.Tensor:
    magnitudes = [
        torch.linalg.vector_norm(env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3), dim=-1)
        for name in contact_sensor_names
    ]
    return (torch.stack(magnitudes, dim=1) > threshold).float().mean(dim=1)


def grasp_point_distance(
    env: ManagerBasedRLEnv,
    std: float,
    palm_cfg: SceneEntityCfg,
    offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    robot: Articulation = env.scene[palm_cfg.name]
    cup: RigidObject = env.scene[object_cfg.name]
    palm_pos = robot.data.body_pos_w[:, palm_cfg.body_ids[0]]
    palm_quat = robot.data.body_quat_w[:, palm_cfg.body_ids[0]]
    local_offset = torch.tensor(offset, device=env.device).expand(env.num_envs, -1)
    grasp_pos = palm_pos + quat_apply(palm_quat, local_offset)
    distance = torch.linalg.vector_norm(cup.data.root_pos_w - grasp_pos, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def fingertips_near_cup(
    env: ManagerBasedRLEnv,
    std: float,
    target_radial_distance: float,
    cup_half_height: float,
    fingertips_cfg: SceneEntityCfg,
    fingertip_offsets: list[tuple[float, float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    robot: Articulation = env.scene[fingertips_cfg.name]
    cup: RigidObject = env.scene[object_cfg.name]
    tip_pos = _fingertip_points_w(robot, fingertips_cfg.body_ids, fingertip_offsets)
    cup_quat = cup.data.root_quat_w[:, None, :].expand(-1, tip_pos.shape[1], -1)
    tip_pos_cup = quat_apply_inverse(cup_quat, tip_pos - cup.data.root_pos_w[:, None, :])
    radial_error = torch.abs(torch.linalg.vector_norm(tip_pos_cup[..., :2], dim=-1) - target_radial_distance)
    axial_error = (torch.abs(tip_pos_cup[..., 2]) - cup_half_height).clamp_min(0.0)
    surface_error = torch.sqrt(torch.square(radial_error) + torch.square(axial_error))
    surface_proximity = 1.0 - torch.tanh(surface_error / std)
    opposing_finger_proximity = torch.max(surface_proximity[:, 1:], dim=1).values
    return surface_proximity[:, 0] * opposing_finger_proximity


def fingertip_opposition(
    env: ManagerBasedRLEnv,
    std: float,
    target_radial_distance: float,
    cup_half_height: float,
    fingertips_cfg: SceneEntityCfg,
    fingertip_offsets: list[tuple[float, float, float]],
    palm_cfg: SceneEntityCfg,
    palm_offset: tuple[float, float, float],
    palm_std: float,
    nominal_xy: tuple[float, float],
    drift_tolerance: float,
    drift_scale: float,
    min_upright: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    robot: Articulation = env.scene[fingertips_cfg.name]
    cup: RigidObject = env.scene[object_cfg.name]
    tip_pos = _fingertip_points_w(robot, fingertips_cfg.body_ids, fingertip_offsets)
    cup_quat = cup.data.root_quat_w[:, None, :].expand(-1, tip_pos.shape[1], -1)
    tip_pos_cup = quat_apply_inverse(cup_quat, tip_pos - cup.data.root_pos_w[:, None, :])

    radial = tip_pos_cup[..., :2]
    radial_distance = torch.linalg.vector_norm(radial, dim=-1)
    radial_error = torch.abs(radial_distance - target_radial_distance)
    axial_error = (torch.abs(tip_pos_cup[..., 2]) - cup_half_height).clamp_min(0.0)
    surface_error = torch.sqrt(torch.square(radial_error) + torch.square(axial_error))
    surface_proximity = 1.0 - torch.tanh(surface_error / std)

    radial_direction = radial / radial_distance.clamp_min(1.0e-6).unsqueeze(-1)
    thumb_direction = radial_direction[:, :1]
    opposition = (1.0 - torch.sum(thumb_direction * radial_direction[:, 1:], dim=-1)) * 0.5
    pair_proximity = surface_proximity[:, :1] * surface_proximity[:, 1:]
    pair_score = pair_proximity * opposition.clamp(0.0, 1.0)

    palm_pos = robot.data.body_pos_w[:, palm_cfg.body_ids[0]]
    palm_quat = robot.data.body_quat_w[:, palm_cfg.body_ids[0]]
    local_offset = torch.tensor(palm_offset, device=env.device).expand(env.num_envs, -1)
    grasp_pos = palm_pos + quat_apply(palm_quat, local_offset)
    palm_distance = torch.linalg.vector_norm(cup.data.root_pos_w - grasp_pos, dim=-1)
    palm_gate = 1.0 - torch.tanh(palm_distance / palm_std)

    local_z = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, -1)
    up_z = quat_apply(cup.data.root_quat_w, local_z)[:, 2]
    upright_gate = ((up_z - min_upright) / (1.0 - min_upright)).clamp(0.0, 1.0)
    cup_xy = cup.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    target_xy = torch.tensor(nominal_xy, device=env.device).expand(env.num_envs, -1)
    drift = torch.linalg.vector_norm(cup_xy - target_xy, dim=-1)
    drift_gate = 1.0 - ((drift - drift_tolerance) / drift_scale).clamp(0.0, 1.0)
    return torch.max(pair_score, dim=1).values * palm_gate * upright_gate * drift_gate


def cup_lift_height(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    threshold: float,
    fingertips_cfg: SceneEntityCfg,
    fingertip_offsets: list[tuple[float, float, float]],
    nominal_xy: tuple[float, float],
    drift_tolerance: float,
    drift_scale: float,
    min_upright: float,
    start_height: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    cup: RigidObject = env.scene[object_cfg.name]
    height = ((cup.data.root_pos_w[:, 2] - start_height) / (target_height - start_height)).clamp(0.0, 1.0)
    grasp_quality = qualified_contact(
        env,
        contact_sensor_names,
        threshold,
        fingertips_cfg,
        fingertip_offsets,
        nominal_xy,
        drift_tolerance,
        drift_scale,
        min_upright,
        object_cfg,
    )
    return height * grasp_quality


def cup_upright(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    threshold: float,
    fingertips_cfg: SceneEntityCfg,
    fingertip_offsets: list[tuple[float, float, float]],
    nominal_xy: tuple[float, float],
    drift_tolerance: float,
    drift_scale: float,
    min_upright: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    cup: RigidObject = env.scene[object_cfg.name]
    local_z = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, -1)
    world_z = quat_apply(cup.data.root_quat_w, local_z)
    upright = world_z[:, 2].clamp(0.0, 1.0)
    grasp_quality = qualified_contact(
        env,
        contact_sensor_names,
        threshold,
        fingertips_cfg,
        fingertip_offsets,
        nominal_xy,
        drift_tolerance,
        drift_scale,
        min_upright,
        object_cfg,
    )
    return upright * (cup.data.root_pos_w[:, 2] > minimal_height) * grasp_quality


def cup_planar_speed(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    cup: RigidObject = env.scene[object_cfg.name]
    return torch.linalg.vector_norm(cup.data.root_lin_vel_w[:, :2], dim=-1)


def cup_planar_drift(
    env: ManagerBasedRLEnv,
    nominal_xy: tuple[float, float],
    tolerance: float,
    scale: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    cup: RigidObject = env.scene[object_cfg.name]
    cup_xy = cup.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    target_xy = torch.tensor(nominal_xy, device=env.device).expand(env.num_envs, -1)
    drift = torch.linalg.vector_norm(cup_xy - target_xy, dim=-1)
    return ((drift - tolerance) / scale).clamp(0.0, 1.0)


def cup_tilt(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    cup: RigidObject = env.scene[object_cfg.name]
    local_z = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, -1)
    world_z = quat_apply(cup.data.root_quat_w, local_z)
    return 1.0 - world_z[:, 2].clamp(0.0, 1.0)


def stable_lift(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    threshold: float,
    fingertips_cfg: SceneEntityCfg,
    fingertip_offsets: list[tuple[float, float, float]],
    nominal_xy: tuple[float, float],
    drift_tolerance: float,
    drift_scale: float,
    min_upright: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    cup: RigidObject = env.scene[object_cfg.name]
    speed = torch.linalg.vector_norm(cup.data.root_lin_vel_w, dim=-1)
    spin = torch.linalg.vector_norm(cup.data.root_ang_vel_w, dim=-1)
    stable = torch.exp(-4.0 * speed - 0.25 * spin)
    local_z = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, -1)
    upright = quat_apply(cup.data.root_quat_w, local_z)[:, 2].clamp(0.0, 1.0)
    grasp_quality = qualified_contact(
        env,
        contact_sensor_names,
        threshold,
        fingertips_cfg,
        fingertip_offsets,
        nominal_xy,
        drift_tolerance,
        drift_scale,
        min_upright,
        object_cfg,
    )
    return stable * upright * (cup.data.root_pos_w[:, 2] > minimal_height) * grasp_quality


def cup_tipped(
    env: ManagerBasedRLEnv,
    minimum_up_axis: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    cup: RigidObject = env.scene[object_cfg.name]
    local_z = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, -1)
    return quat_apply(cup.data.root_quat_w, local_z)[:, 2] < minimum_up_axis


def joint_velocity_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    velocity = robot.data.joint_vel[:, asset_cfg.joint_ids].clamp(-100.0, 100.0)
    return torch.sum(torch.square(velocity), dim=1)
