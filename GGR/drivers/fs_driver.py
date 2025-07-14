import argparse

from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import decode_config

def main(args):
    config = decode_config(args.config)

    command = f'python -m frame_sampling {config["ia_filtered_out_path"]} {config["fs_stage1_out_path"]} {config["fs_out_path"]}'

    logger = setup_logging(config["fs_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the frame sampling component of the pipeline. Performs non-maximum supression to select ideal annotations across a set of tracked annotations."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
