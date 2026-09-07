#!/usr/bin/env python3
import os
from pathlib import Path


source_path = Path(__file__).with_name("r1_a7_o6_live_ik_preview.py")
source = source_path.read_text(encoding="utf-8")
damping = float(os.environ.get("R1_AB_DAMPING", "4.0"))
mode = os.environ.get("R1_AB_MODE", "baseline")
source = source.replace(
    "                damping=4.0,\n",
    f"                damping={damping!r},\n",
    1,
)
if mode == "settle_hold":
    source = source.replace(
        "    condition_max_window = 0.0\n\n    print(\"[R1-A7 O6 LIVE] fixed-base CAD mount; simulation only; DDS/network/serial are not used\")",
        "    condition_max_window = 0.0\n"
        "    previous_active_arm_offset = active_arm_offset.copy()\n"
        "    held_left_desired = left_hold.clone()\n"
        "    held_right_desired = right_hold.clone()\n"
        "    ik_target_held = False\n\n"
        "    print(\"[R1-A7 O6 LIVE] fixed-base CAD mount; simulation only; DDS/network/serial are not used\")",
        1,
    )
    source = source.replace(
        "            active_arm_offset = rate_limit_offset(active_arm_offset, arm_offset)\n",
        "            active_arm_offset = rate_limit_offset(active_arm_offset, arm_offset)\n"
        "            if np.max(np.abs(active_arm_offset - previous_active_arm_offset)) > 1.0e-8:\n"
        "                ik_target_held = False\n"
        "            previous_active_arm_offset = active_arm_offset.copy()\n",
        1,
    )
    source = source.replace(
        "                sigma_min_window = min(sigma_min_window, left_sigma_min, right_sigma_min)\n"
        "                condition_max_window = max(condition_max_window, left_condition, right_condition)\n"
        "                arm_tracking = True\n",
        "                sigma_min_window = min(sigma_min_window, left_sigma_min, right_sigma_min)\n"
        "                condition_max_window = max(condition_max_window, left_condition, right_condition)\n"
        "                if ik_target_held:\n"
        "                    left_desired = held_left_desired\n"
        "                    right_desired = held_right_desired\n"
        "                else:\n"
        "                    held_left_desired = left_desired.clone()\n"
        "                    held_right_desired = right_desired.clone()\n"
        "                    current_left_pos, _ = body_pose_in_root(robot, left_body_id)\n"
        "                    current_right_pos, _ = body_pose_in_root(robot, right_body_id)\n"
        "                    command_settled = np.max(np.abs(active_arm_offset - arm_offset)) <= 1.0e-8\n"
        "                    cartesian_settled = (\n"
        "                        float(torch.linalg.vector_norm(left_goal_pos - current_left_pos)) <= 0.001\n"
        "                        and float(torch.linalg.vector_norm(right_goal_pos - current_right_pos)) <= 0.001\n"
        "                    )\n"
        "                    if command_settled and cartesian_settled:\n"
        "                        ik_target_held = True\n"
        "                arm_tracking = True\n",
        1,
    )
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
