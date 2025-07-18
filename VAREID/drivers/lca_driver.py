import argparse

from VAREID.util.io.format_funcs import load_config
from VAREID.util.io.logging import log_subprocess, setup_logging
from VAREID.util.io.workflow_funcs import build_config, decode_config

def get_inputs(config):
    inputs = [config["mid_out_path"]]

    if config["data_video"]:
        inputs.append(config["fs_out_path"])
    else:
        inputs.append(config["ia_filtered_out_path"])

    return inputs


def get_outputs(config):
    if config["lca_separate_viewpoints"]:
        outputs = [config["post_left_in_path"], config["post_right_in_path"]]
    else:
        outputs = [config["lca_out_path"]]

    return outputs


def main(args):
    # SELECT THE CORRECT CONFIG
    if args.config:
        config = decode_config(args.config)
    else:
        config = build_config(load_config(args.config_path))

    input = config["fs_out_path"] if config["data_video"] else config["ia_filtered_out_path"]
    video_flag = "--video" if config["data_video"] else ""
    sv_flag = "--separate_viewpoints" if config["lca_separate_viewpoints"] else ""

    command = f'python -u -m VAREID.algo.lca.lca {input} {config["mid_out_path"]} {config["lca_dir"]} {config["lca_out_prefix"]} {config["lca_out_suffix"]} {config["lca_subunit_logs"]} {config["lca_logs"]} {video_flag} {sv_flag}'

    logger = setup_logging(config["lca_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the LCA component of the pipeline. Clusters annotations."
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
    