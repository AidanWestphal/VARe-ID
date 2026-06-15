import argparse

from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.logging import log_subprocess, setup_logging
from VAREID.libraries.io.workflow_funcs import build_config, decode_config


def get_inputs(config):
    return [config["inter_lca_out_path"]]


def get_outputs(config):
    return [config["lca_out_path"]]


def main(args):
    # SELECT THE CORRECT CONFIG
    if args.config:
        config = decode_config(args.config)
    else:
        config = build_config(load_config(args.config_path))

    input = config["inter_lca_out_path"]

    command = f'python -u -m VAREID.algo.forward_clustering.forward_clustering {input} {config["lca_out_path"]}'

    logger = setup_logging(config["forward_clustering_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the forward clustering component of the pipeline. Propagates inter_cluster_id from representatives to non-representatives."
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