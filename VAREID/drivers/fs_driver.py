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
    
    if config["fs_stage1_out_path"]:
        json_stage1 = "--json_stage1 " + config["fs_stage1_out_path"]
    else:
        json_stage1 = ""

    command = f'python -u -m VAREID.algo.frame_sampling.frame_sampling {config["ia_filtered_out_path"]} {config["fs_out_path"]} {json_stage1}'

    logger = setup_logging(config["fs_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the frame sampling component of the pipeline. Performs non-maximum supression to select ideal annotations across a set of tracked annotations."
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
    