import argparse
import os.path

from VAREID.libraries.constants import GREVYS_ZEBRA
from VAREID.libraries.db.table import ImageTable

from VAREID.libraries.ggr_funcs import extrapolate_ggr_gps

from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.logging import setup_logging, log_subprocess
from VAREID.libraries.io.workflow_funcs import build_config, decode_config

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Import database of GGR images and extrapolate GPS coordinates for images without GPS')
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
    
    if args.config:
        config = decode_config(args.config)
    else:
        config = build_config(load_config(args.config_path))

    # TODO: Update to use import_image_driver.py probably
    # Load image data from json when applicable
    if os.path.isfile(config["data_dir_in"]) and os.path.getsize(config["data_dir_in"]) != 0:
        imgtable = ImageTable(os.path.dirname(config["data_dir_in"]))
        imgtable.import_from_json(config["data_dir_in"])
    else:
        print("Unable to import image data... (exiting)")
        exit(-1)
    
    # Add images to database
    skipped_gid_list = extrapolate_ggr_gps(imgtable, doctest_mode=False)
    imgtable.export_to_json(config["data_dir_out"])