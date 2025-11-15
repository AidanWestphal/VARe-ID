import argparse

from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.logging import log_subprocess, setup_logging
from VAREID.libraries.io.workflow_funcs import build_config, decode_config

def get_inputs(config):
    inputs = [config["mid_out_path"]]

    if config["data_video"]:
        inputs.append(config["fs_out_path"])
    else:
        inputs.append(config["ia_filtered_out_path"])

    return inputs


def get_outputs(config):
    if config.get("lca_separate_by_fields"):
        # For multi-field separation, we don't know exact outputs without knowing field values
        # Return base path - actual outputs will be generated based on field combinations
        outputs = [config["lca_dir"]]  # Base directory contains all outputs
    elif config.get("lca_separate_viewpoints"):
        # Legacy viewpoint separation
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

    input = config["fs_out_path"] if config["data_video"] else config["eg_out_path"]
    video_flag = "--video" if config["data_video"] else ""
    
    # Handle field separation (new) vs viewpoint separation (legacy)
    if config.get("lca_separate_by_fields"):
        # New multi-field separation
        fields = config["lca_separate_by_fields"]
        if isinstance(fields, str):
            fields = fields.split()  # Convert string to list if needed
        separation_flag = f"--separate_by_fields {' '.join(fields)}"
    elif config.get("lca_separate_viewpoints"):
        # Legacy viewpoint separation
        separation_flag = "--separate_viewpoints"
    else:
        separation_flag = ""

    command = f'python -u -m VAREID.algo.lca.lca {input} {config["mid_out_path"]} {config["lca_dir"]} {config["lca_out_prefix"]} {config["lca_out_suffix"]} {config["lca_subunit_logs"]} {config["lca_logs"]} {video_flag} {separation_flag}'

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
    