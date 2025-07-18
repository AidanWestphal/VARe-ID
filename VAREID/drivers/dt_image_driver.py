import argparse

from VAREID.util.io.format_funcs import load_config
from VAREID.util.io.logging import log_subprocess, setup_logging
from VAREID.util.io.workflow_funcs import build_config, decode_config

def main(args):
    # SELECT THE CORRECT CONFIG
    if args.config:
        config = decode_config(args.config)
    else:
        config = build_config(load_config(args.config_path))
    
    # CHECK FOR "NONE" GT FIELDS AND PROPERLY BUILD THOSE TERMS IN COMMAND
    if config["dt_filtered_out_path"]:
        gt_filtered_annots = "--gt_filtered_annots " + config["dt_filtered_out_path"]
    else:
        gt_filtered_annots = ""

    if config["dt_gt_path"]:
        gt_path = "--gt_path " + config["dt_gt_path"]
    else:
        gt_path = ""

    command = f'python -u -m VAREID.algo.detection.image_detector {config["image_out_path"]} {config["dt_dir"]} {config["dt_model"]} {config["dt_image_out_path"]} {gt_filtered_annots} {gt_path}'

    logger = setup_logging(config["dt_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the detection component of the pipeline. Generates base annotations from the set of images (or video frames)."
    )
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--config",
        type=str,
        default=None,
        help="The built config file as a base64 encoded string. Config file MUST be structured like config.yaml!",
    )
    group.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="A path to the config file to load. Config file MUST be structured like config.yaml!",
    )
    args = parser.parse_args()

    main(args)
