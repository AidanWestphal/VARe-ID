import argparse
import os.path

from VAREID.util.constants import GREVYS_ZEBRA
from VAREID.util.db.table import ImageTable

from VAREID.util.ggr_funcs import extrapolate_ggr_gps

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Import database of GGR images and extrapolate GPS coordinates for images without GPS')
    parser.add_argument('in_csv_path', type=str, help='The image data json file to import from')
    parser.add_argument('out_csv_path', type=str, help='The full path to the .json file to store image data in')
    args = parser.parse_args()

    # Load image data from json when applicable
    if os.path.isfile(args.in_csv_path) and os.path.getsize(args.in_csv_path) != 0:
        imgtable = ImageTable(os.path.dirname(args.in_csv_path))
        imgtable.import_from_json(args.in_csv_path)
    else:
        print("Unable to import image data... (exiting)")
        exit(-1)
    
    # Add images to database
    skipped_gid_list = extrapolate_ggr_gps(imgtable, doctest_mode=False)
    imgtable.export_to_json(args.out_csv_path)