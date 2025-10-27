import argparse
import os.path

from VAREID.libraries.constants import GREVYS_ZEBRA
from VAREID.libraries.db.table import ImageTable

from VAREID.libraries.ggr_funcs import extrapolate_ggr_gps

from VAREID.libraries.io.format_funcs import load_config
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

    imgtable = ImageTable(config["data_dir_out"])
    
    json_path = os.path.join(config["data_dir_out"], config["image_out_file"])
    
    if not os.path.exists(json_path):
        print(f"{json_path} does not exist.")
        exit(-1)
    if not json_path[-5] == ".json":
        print(f"Extrapolate ggr gps needs a .json file")
        print(f"{json_path} is not a JSON file")
        exit(-1)
    
    imgtable.import_from_json(json_path)

    # Add images to database
    skipped_gid_list = extrapolate_ggr_gps(imgtable, doctest_mode=False)
    imgtable.export_to_json(json_path)