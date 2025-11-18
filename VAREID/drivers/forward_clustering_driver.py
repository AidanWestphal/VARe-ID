import argparse

from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.logging import log_subprocess, setup_logging
from VAREID.libraries.io.workflow_funcs import build_config, decode_config


def get_inputs(config):
    """Get input paths based on configuration."""
    # Input should be the representative selection output with representative field
    return [config.get("rep_out_path", config["lca_out_path"].replace("lca_", "rep_"))]


def get_outputs(config):
    """Get output paths based on configuration."""
    # Output path for annotations with intra_cluster_id assigned
    return [config.get("fc_out_path", config.get("rep_out_path", config["lca_out_path"]).replace("rep_", "fc_"))]


def main(args):
    # SELECT THE CORRECT CONFIG
    if args.config:
        config = decode_config(args.config)
    else:
        config = build_config(load_config(args.config_path))

    # Get paths
    input_path = get_inputs(config)[0]
    output_path = get_outputs(config)[0]

    # Get the cluster field name from config (default to 'inter_cluster_id')
    cluster_field = config.get("fc_cluster_field", "inter_cluster_id")

    # Get the intra cluster field name from config (default to 'intra_cluster_id')
    intra_field = config.get("fc_intra_field", "intra_cluster_id")

    # Get the representative field name from config (default to 'representative')
    rep_field = config.get("fc_rep_field", "representative")

    # Build command to run the forward clustering algorithm
    command = f'python -u -m VAREID.algo.forward_clustering.forward_clustering {input_path} {output_path} --cluster_field {cluster_field} --intra_field {intra_field} --rep_field {rep_field}'

    # Setup logging
    log_path = config.get("fc_logs", "logs/forward_clustering.log")
    logger = setup_logging(log_path)
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the forward clustering component of the pipeline. Assigns intra_cluster_id to non-representative annotations based on their representative annotation."
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