import torch

from isaaclab.envs.mdp.actions.actions_cfg import EMAJointPositionToLimitsActionCfg
from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


MIMIC_JOINTS = {
    "rh_thumb_ip": ("rh_thumb_cmc_pitch", 1.86),
    "rh_index_dip": ("rh_index_mcp_pitch", 0.89),
    "rh_middle_dip": ("rh_middle_mcp_pitch", 0.89),
    "rh_ring_dip": ("rh_ring_mcp_pitch", 0.89),
    "rh_pinky_dip": ("rh_pinky_mcp_pitch", 0.89),
}


class O6MimicEMAJointPositionAction(EMAJointPositionToLimitsAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._mimic_joint_ids, _ = self._asset.find_joints(list(MIMIC_JOINTS), preserve_order=True)
        self._source_action_ids = [self._joint_names.index(source) for source, _ in MIMIC_JOINTS.values()]
        self._mimic_ratios = self._raw_actions.new_tensor([ratio for _, ratio in MIMIC_JOINTS.values()])
        self._hand_action_mask = self._raw_actions.new_tensor(
            [name.startswith("rh_") for name in self._joint_names], dtype=torch.bool
        )

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        normalized = (self._raw_actions * self._scale).clamp(-1.0, 1.0)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        default = self._asset.data.default_joint_pos[:, self._joint_ids]
        default_centered_targets = torch.where(
            normalized >= 0.0,
            default + normalized * (limits[..., 1] - default),
            default + normalized * (default - limits[..., 0]),
        )
        full_range_targets = limits[..., 0] + 0.5 * (normalized + 1.0) * (limits[..., 1] - limits[..., 0])
        targets = torch.where(self._hand_action_mask, full_range_targets, default_centered_targets)
        targets = self._alpha * targets + (1.0 - self._alpha) * self._prev_applied_actions
        self._processed_actions[:] = targets.clamp(limits[..., 0], limits[..., 1])
        self._prev_applied_actions[:] = self._processed_actions

    def apply_actions(self):
        super().apply_actions()
        targets = self.processed_actions[:, self._source_action_ids] * self._mimic_ratios
        limits = self._asset.data.soft_joint_pos_limits[:, self._mimic_joint_ids]
        targets = targets.clamp(limits[..., 0], limits[..., 1])
        self._asset.set_joint_position_target(targets, joint_ids=self._mimic_joint_ids)


@configclass
class O6MimicEMAJointPositionActionCfg(EMAJointPositionToLimitsActionCfg):
    class_type: type[ActionTerm] = O6MimicEMAJointPositionAction
