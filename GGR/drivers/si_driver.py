import argparse
import subprocess

from GGR.drivers.workflow_funcs import decode_config

def get_inputs(config):
    return [config["dt_video_out_path"]] if config["data_video"] else [config["dt_image_out_path"]]


def main(args):
    config = decode_config(args.config)

    try:
        subprocess.run(
            f'python -m species_identifier {config["dt_out_path"]} {config["si_dir"]} {config["si_out_path"]} &> {config["si_logs"]}',
            shell=True, text=True, check=True
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the species identification component of the pipeline. Classifies annotations by species (grevys, plains, or neither)."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
