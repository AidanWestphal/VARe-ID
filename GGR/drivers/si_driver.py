import argparse

from GGR.util.io.format_funcs import load_config
from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import build_config, decode_config

def get_inputs(config):
    return [config["dt_video_out_path"]] if config["data_video"] else [config["dt_image_out_path"]]


def main(args):
    # SELECT THE CORRECT CONFIG
    if args.config:
        config = decode_config(args.config)
    else:
        config = build_config(load_config(args.config_path))
        
    input = config["dt_video_out_path"] if config["data_video"] else config["dt_image_out_path"]

    # python -m GGR.algo.species_identification.species_identifier /fs/ess/PAS2136/ggr_data/results/GGR2020_subset_refactored/detector/img_annots.json /fs/ess/PAS2136/ggr_data/results/GGR2020_subset_refactored/species_identifier /fs/ess/PAS2136/ggr_data/results/GGR2020_subset_refactored/species_identifier/si_annots.json
    command = f'python -u -m GGR.algo.species_identification.species_identifier {input} {config["si_dir"]} {config["si_out_path"]}'
    logger = setup_logging(config["si_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the species identification component of the pipeline. Classifies annotations by species (grevys, plains, or neither)."
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
    