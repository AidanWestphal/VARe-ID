import argparse
import json
import os
import sys

from shapely import Point

from VAREID.libraries.ggr_funcs import get_ggr_polygons, match_point_to_poly


def filter_by_county(in_path, out_path, counties_json, target_county):
    with open(in_path, "r") as f:
        data = json.load(f)

    poly_dict_c = get_ggr_polygons(filepath=counties_json, c_or_lt=0, invert=True)
    target_polys = [p for p, name in poly_dict_c.items() if name == target_county]
    if not target_polys:
        print(f"County '{target_county}' not found. Available: {sorted(poly_dict_c.values())}")
        sys.exit(1)

    images = data.get("images", [])
    kept_images = []
    poly_prev = None
    for img in images:
        lat, lon = img.get("gps_lat"), img.get("gps_lon")
        if lat is None or lon is None or (lat, lon) == (-1, -1):
            continue
        coord = Point((lat, lon))
        poly = match_point_to_poly(coord, poly_prev, target_polys)
        if poly is not None:
            kept_images.append(img)
            poly_prev = poly

    kept_uuids = {img["uuid"] for img in kept_images}
    kept_annotations = [a for a in data.get("annotations", []) if a.get("image_uuid") in kept_uuids]

    out = {
        "categories": data.get("categories", []),
        "images": kept_images,
        "annotations": kept_annotations,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"Input images:       {len(images)}")
    print(f"Kept images:        {len(kept_images)}  (county = {target_county})")
    print(f"Input annotations:  {len(data.get('annotations', []))}")
    print(f"Kept annotations:   {len(kept_annotations)}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a VARe-ID id_regions-style json to a single GGR county by GPS.")
    parser.add_argument("--in_json_path", required=True)
    parser.add_argument("--out_json_path", required=True)
    parser.add_argument("--counties_file", required=True)
    parser.add_argument("--county", required=True, help="County name to keep (e.g. 'Laikipia')")
    args = parser.parse_args()
    filter_by_county(args.in_json_path, args.out_json_path, args.counties_file, args.county)
