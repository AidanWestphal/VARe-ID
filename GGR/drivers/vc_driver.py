import argparse

from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import decode_config

def main(args):
    config = decode_config(args.config)

    command = f'python -m viewpoint_classifier {config["si_out_path"]} {config["vc_model_path"]} {config["vc_out_path"]}'

    logger = setup_logging(config["vc_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the viewpoint classification component of the pipeline. Classifies an annotation by its viewpoint (up, left, right, front, back, or any combination of those)."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
