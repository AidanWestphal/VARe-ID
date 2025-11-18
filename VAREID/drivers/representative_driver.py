import argparse

from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.logging import log_subprocess, setup_logging
from VAREID.libraries.io.workflow_funcs import build_config, decode_config


def get_inputs(config):
    """Get input paths based on configuration."""
    # Input should be the LCA output path with inter_cluster_id assignments
    return [config["lca_out_path"]]


def get_outputs(config):
    """Get output paths based on configuration."""
    # Output path for annotations with representative field added
    return [config.get("rep_out_path", config["lca_out_path"].replace("lca_", "rep_"))]


def main(args):
    # SELECT THE CORRECT CONFIG
    if args.config:
        config = decode_config(args.config)
    else:
        config = build_config(load_config(args.config_path))

    # Get paths
    input_path = get_inputs(config)[0]
    output_path = get_outputs(config)[0]

    # Get the IA score field name from config (default to 'ia_score')
    ia_field = config.get("rep_ia_field", "ia_score")

    # Get the cluster field name from config (default to 'inter_cluster_id')
    cluster_field = config.get("rep_cluster_field", "inter_cluster_id")

    # Build command to run the representative selection algorithm
    command = f'python -u -m VAREID.algo.representative_selection.representative_selection {input_path} {output_path} --ia_field {ia_field} --cluster_field {cluster_field}'

    # Setup logging
    log_path = config.get("rep_logs", "logs/representative.log")
    logger = setup_logging(log_path)
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the representative selection component of the pipeline. Selects the annotation with highest IA score as representative for each cluster."
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