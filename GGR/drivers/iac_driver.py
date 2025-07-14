import argparse

from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import decode_config

def main(args):
    config = decode_config(args.config)

    command = f'python -m GGR.algo.ia_classification.IA_classifier {config["vc_out_path"]} {config["ia_model_path"]} {config["ia_out_path"]}'

    logger = setup_logging(config["ia_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the identifiable annotation classification component of the pipeline. Classifies an annotation by its quality/ability to be identified."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
