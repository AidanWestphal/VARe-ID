import argparse

from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import decode_config

def main(args):
    config = decode_config(args.config)
    
    # CHECK FOR "NONE" GT FIELDS AND PROPERLY BUILD THOSE TERMS IN COMMAND
    if config["dt_filtered_out_path"]:
        gt_filtered_annots = "--gt_filtered_annots " + config["dt_filtered_out_path"]
    else:
        gt_filtered_annots = ""

    if config["dt_gt_path"]:
        gt_path = "--gt_path " + config["dt_gt_path"]
    else:
        gt_path = ""

    command = f'python -m GGR.algo.detection.image_detector {config["image_out_path"]} {config["dt_dir"]} {config["dt_model"]} {config["dt_image_out_path"]} {gt_filtered_annots} {gt_path}'

    logger = setup_logging(config["dt_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the detection component of the pipeline. Generates base annotations from the set of images (or video frames)."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
