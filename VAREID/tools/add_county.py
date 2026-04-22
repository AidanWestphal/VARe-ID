import argparse
import os.path
import json
import yaml
import sys
import geopandas  # type: ignore
import json
import os
from qreader import QReader
from shapely import MultiPolygon, Polygon, Point
from VAREID.libraries.ggr_funcs import *

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Import database of GGR images and match each image with GPS to a county and land holding')
    parser.add_argument('--in_json_path', type=str, help='The image data json file to import from')
    parser.add_argument('--out_json_path', type=str, help='The full path to the .json file to store image data in')
    args = parser.parse_args()
    
    # Load image data from json when applicable
    if os.path.isfile(args.in_json_path) and os.path.getsize(args.in_json_path) != 0:
        with open(args.in_json_path, 'r') as f:
            data = json.load(f)
        img_data = data['images']
    else:
        print("Unable to import image data... (exiting)")
        exit(-1)
    
    poly_dict_c = get_ggr_polygons(filepath="VAREID/ggr_counties.json", 
                                   c_or_lt=0, invert=True)
    poly_dict_lt = get_ggr_polygons(filepath="VAREID/ggr_landtenures.json", 
                                    c_or_lt=1, invert=True)
    c_prev = None
    lt_prev = None
    for info in img_data:
        lat = info['gps_lat']
        lon = info['gps_lon']
        coord = Point((lat, lon))
        
        c_cur = match_point_to_poly(coord, c_prev, poly_dict_c.keys())
        lt_cur = match_point_to_poly(coord, lt_prev, poly_dict_lt.keys())
        c_name = poly_dict_c[c_cur] if c_cur else None
        lt_name = poly_dict_lt[lt_cur] if lt_cur else None
        
        info['county'] = c_name
        info['land tenure'] = lt_name
        
        c_prev = c_cur
        lt_prev = lt_cur

data['images'] = img_data

with open(args.out_json_path, 'w') as f:
    json.dump(data, f)