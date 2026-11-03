import gymnasium as gym

from . import agents
from .lift_cup_r1_a7_o6_env_cfg import LiftCupR1A7O6EnvCfg, LiftCupR1A7O6EnvCfg_PLAY


gym.register(
    id="Isaac-Lift-Cup-R1A7-O6-Right-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": LiftCupR1A7O6EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCupPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cup-R1A7-O6-Right-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": LiftCupR1A7O6EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCupPPORunnerCfg",
    },
    disable_env_checker=True,
)
