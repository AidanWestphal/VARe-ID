import argparse

from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import decode_config

def get_inputs(config):
    return [config["dt_video_out_path"]] if config["data_video"] else [config["dt_image_out_path"]]


def main(args):
    config = decode_config(args.config)

    command = f'python -m species_identifier {config["dt_out_path"]} {config["si_dir"]} {config["si_out_path"]}'

    logger = setup_logging(config["si_logs"])
    log_subprocess(command, logger)


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
