import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    input_path = os.path.abspath(args_cli.input)
    output_path = os.path.abspath(args_cli.output)
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=input_path,
            usd_dir=os.path.dirname(output_path),
            usd_file_name=os.path.basename(output_path),
            fix_base=True,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=False,
            self_collision=True,
            make_instanceable=True,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=60.0, damping=3.0),
            ),
        )
    )
    print(converter.usd_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
