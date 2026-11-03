import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from robots.unitree import R1A7_O6_BASE_FIX_CFG, R1A7_RIGHT_ARM_JOINTS, R1A7_RIGHT_O6_ACTIVE_JOINTS

from . import mdp


CONTROLLED_JOINTS = R1A7_RIGHT_ARM_JOINTS + R1A7_RIGHT_O6_ACTIVE_JOINTS
FINGERTIP_BODIES = [
    "rh_thumb_distal",
    "rh_index_distal",
    "rh_middle_distal",
    "rh_ring_distal",
    "rh_pinky_distal",
]
FINGERTIP_OFFSETS = [
    (-0.0040, 0.0, 0.0440),
    (0.0065, 0.0, 0.0340),
    (0.0065, 0.0, 0.0340),
    (0.0065, 0.0, 0.0340),
    (0.0065, 0.0, 0.0340),
]
CONTACT_SENSOR_NAMES = [f"{name}_cup_contact" for name in FINGERTIP_BODIES]
THUMB_CONTACT_PROBE_BODIES = ["rh_thumb_metacarpals", "rh_thumb_metacarpals_base2"]
THUMB_CONTACT_PROBE_SENSOR_NAMES = [f"{name}_cup_contact" for name in THUMB_CONTACT_PROBE_BODIES]
NON_RIGHT_HAND_BODY_PATTERN = (
    "(pelvis_link|waist_yaw_link|left_.*_link|right_(shoulder|elbow|wrist).*_link|"
    "head_.*_link|lh_.*|right_o6_adapter_link)"
)


@configclass
class LiftCupSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = R1A7_O6_BASE_FIX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, -0.30, 0.80)),
        spawn=sim_utils.CuboidCfg(
            size=(0.65, 0.65, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.28, 0.32)),
        ),
    )

    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.566, -0.320, 0.87), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.08, 0.05), roughness=0.45),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    rh_thumb_distal_cup_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_thumb_distal",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"],
        update_period=0.0,
    )
    rh_index_distal_cup_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_index_distal",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"],
        update_period=0.0,
    )
    rh_middle_distal_cup_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_middle_distal",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"],
        update_period=0.0,
    )
    rh_ring_distal_cup_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_ring_distal",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"],
        update_period=0.0,
    )
    rh_pinky_distal_cup_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_pinky_distal",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"],
        update_period=0.0,
    )
    rh_thumb_metacarpals_cup_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_thumb_metacarpals",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"],
        update_period=0.0,
    )
    rh_thumb_metacarpals_base2_cup_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_thumb_metacarpals_base2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"],
        update_period=0.0,
    )
    robot_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0)


@configclass
class ActionsCfg:
    joint_position = mdp.O6MimicEMAJointPositionActionCfg(
        asset_name="robot",
        joint_names=CONTROLLED_JOINTS,
        scale=1.0,
        rescale_to_limits=True,
        preserve_order=True,
        alpha={"right_.*": 0.10, "rh_.*": 0.20},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
            scale=0.1,
        )
        cup_pose = ObsTerm(func=mdp.object_pose_in_robot_root)
        cup_velocity = ObsTerm(func=mdp.object_velocity_in_robot_root, scale=0.2)
        fingertips = ObsTerm(
            func=mdp.body_positions_in_robot_root,
            params={
                "body_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES, preserve_order=True),
            },
        )
        grasp_delta = ObsTerm(
            func=mdp.grasp_point_to_object,
            params={
                "palm_cfg": SceneEntityCfg("robot", body_names=["rh_hand_base_link"]),
                "offset": (0.0, 0.0, 0.11),
            },
        )
        contacts = ObsTerm(
            func=mdp.contact_forces_in_robot_root,
            params={"contact_sensor_names": CONTACT_SENSOR_NAMES},
            clip=(-20.0, 20.0),
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    reset_cup = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup"),
        },
    )


@configclass
class RewardsCfg:
    reach = RewTerm(
        func=mdp.grasp_point_distance,
        weight=2.0,
        params={
            "std": 0.12,
            "palm_cfg": SceneEntityCfg("robot", body_names=["rh_hand_base_link"]),
            "offset": (0.0, 0.0, 0.11),
        },
    )
    fingertips = RewTerm(
        func=mdp.fingertips_near_cup,
        weight=4.0,
        params={
            "std": 0.08,
            "target_radial_distance": 0.04,
            "cup_half_height": 0.05,
            "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES, preserve_order=True),
            "fingertip_offsets": FINGERTIP_OFFSETS,
        },
    )
    opposition = RewTerm(
        func=mdp.fingertip_opposition,
        weight=12.0,
        params={
            "std": 0.04,
            "target_radial_distance": 0.04,
            "cup_half_height": 0.05,
            "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES, preserve_order=True),
            "fingertip_offsets": FINGERTIP_OFFSETS,
            "palm_cfg": SceneEntityCfg("robot", body_names=["rh_hand_base_link"]),
            "palm_offset": (0.0, 0.0, 0.11),
            "palm_std": 0.12,
            "nominal_xy": (0.566, -0.320),
            "drift_tolerance": 0.05,
            "drift_scale": 0.05,
            "min_upright": 0.8660254,
        },
    )
    fingertip_contact = RewTerm(
        func=mdp.fingertip_contact_count,
        weight=4.0,
        params={"contact_sensor_names": CONTACT_SENSOR_NAMES, "threshold": 0.2},
    )
    contact = RewTerm(
        func=mdp.qualified_contact,
        weight=12.0,
        params={
            "contact_sensor_names": CONTACT_SENSOR_NAMES,
            "threshold": 0.2,
            "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES, preserve_order=True),
            "fingertip_offsets": FINGERTIP_OFFSETS,
            "nominal_xy": (0.566, -0.320),
            "drift_tolerance": 0.05,
            "drift_scale": 0.05,
            "min_upright": 0.8660254,
        },
    )
    lift = RewTerm(
        func=mdp.cup_lift_height,
        weight=80.0,
        params={
            "contact_sensor_names": CONTACT_SENSOR_NAMES,
            "threshold": 0.2,
            "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES, preserve_order=True),
            "fingertip_offsets": FINGERTIP_OFFSETS,
            "nominal_xy": (0.566, -0.320),
            "drift_tolerance": 0.05,
            "drift_scale": 0.05,
            "min_upright": 0.8660254,
            "start_height": 0.87,
            "target_height": 0.97,
        },
    )
    upright = RewTerm(
        func=mdp.cup_upright,
        weight=8.0,
        params={
            "contact_sensor_names": CONTACT_SENSOR_NAMES,
            "threshold": 0.2,
            "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES, preserve_order=True),
            "fingertip_offsets": FINGERTIP_OFFSETS,
            "nominal_xy": (0.566, -0.320),
            "drift_tolerance": 0.05,
            "drift_scale": 0.05,
            "min_upright": 0.8660254,
            "minimal_height": 0.88,
        },
    )
    stable = RewTerm(
        func=mdp.stable_lift,
        weight=24.0,
        params={
            "contact_sensor_names": CONTACT_SENSOR_NAMES,
            "threshold": 0.2,
            "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES, preserve_order=True),
            "fingertip_offsets": FINGERTIP_OFFSETS,
            "nominal_xy": (0.566, -0.320),
            "drift_tolerance": 0.05,
            "drift_scale": 0.05,
            "min_upright": 0.8660254,
            "minimal_height": 0.88,
        },
    )
    goal = RewTerm(
        func=mdp.stable_lift,
        weight=40.0,
        params={
            "contact_sensor_names": CONTACT_SENSOR_NAMES,
            "threshold": 0.2,
            "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES, preserve_order=True),
            "fingertip_offsets": FINGERTIP_OFFSETS,
            "nominal_xy": (0.566, -0.320),
            "drift_tolerance": 0.05,
            "drift_scale": 0.05,
            "min_upright": 0.8660254,
            "minimal_height": 0.92,
        },
    )
    planar_speed = RewTerm(func=mdp.cup_planar_speed, weight=-0.5)
    planar_drift = RewTerm(
        func=mdp.cup_planar_drift,
        weight=-8.0,
        params={"nominal_xy": (0.566, -0.320), "tolerance": 0.05, "scale": 0.05},
    )
    tilt = RewTerm(func=mdp.cup_tilt, weight=-4.0)
    tipped = RewTerm(func=mdp.cup_tipped, weight=-100.0, params={"minimum_up_axis": 0.5})
    non_right_hand_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("robot_contacts", body_names=[NON_RIGHT_HAND_BODY_PATTERN]),
            "threshold": 1.0,
        },
    )
    dropped = RewTerm(
        func=mdp.root_height_below_minimum,
        weight=-100.0,
        params={"minimum_height": 0.76, "asset_cfg": SceneEntityCfg("cup")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-3)
    joint_velocity = RewTerm(
        func=mdp.joint_velocity_l2_clamped,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cup_tipped = DoneTerm(func=mdp.cup_tipped, params={"minimum_up_axis": 0.5})
    cup_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.76, "asset_cfg": SceneEntityCfg("cup")},
    )


@configclass
class LiftCupR1A7O6EnvCfg(ManagerBasedRLEnvCfg):
    scene: LiftCupSceneCfg = LiftCupSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 4 * 1024 * 1024


@configclass
class LiftCupR1A7O6EnvCfg_PLAY(LiftCupR1A7O6EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.0
        self.viewer.eye = (1.4, -1.4, 1.5)
        self.viewer.lookat = (0.45, -0.30, 0.90)
