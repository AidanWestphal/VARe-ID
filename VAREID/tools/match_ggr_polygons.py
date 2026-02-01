import argparse
import os.path
import json
import yaml
import sys

from VAREID.libraries.constants import GREVYS_ZEBRA
from VAREID.libraries.db.table import ImageTable

from VAREID.libraries.ggr_funcs import match_ggr_polygons

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Import database of GGR images and match each image with GPS to a county and land holding')
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML configuration file')
    parser.add_argument('--in_json_path', type=str, help='The image data json file to import from')
    parser.add_argument('--out_json_path', type=str, help='The full path to the .json file to store image data in')
    parser.add_argument('--proj_path', type=str, help='Path to VARe-ID project directory')
    args = parser.parse_args()

    config_data = {}

    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Config file '{args.config}' not found.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            sys.exit(1)
    
    if config_data:
        for key, value in config_data.items():
            if value is not None and getattr(args, key, None) is None:
                 setattr(args, key, value)

    # Load image data from json when applicable
    if os.path.isfile(args.in_json_path) and os.path.getsize(args.in_json_path) != 0:
        imgtable = ImageTable(os.path.dirname(args.in_json_path))
        imgtable.import_from_json(args.in_json_path)
    else:
        print("Unable to import image data... (exiting)")
        exit(-1)
    
    # Write location data to csv
    c_lt_by_uuids = match_ggr_polygons(imgtable, args.proj_path)
    print("Exporting to json...")
    with open(args.out_json_path, "w") as outfile:
        json.dump(c_lt_by_uuids, outfile, indent=4)

    print(f"\t...exported counties/land holdings for {len(c_lt_by_uuids)} image records")