'''
## Annotation Visualization Tool

The following tool helps visualize annotations generated from the pipeline by overlaying them onto images.

Import data under the `annots` parameter. This should be a path to the post CA classifier annotation file. Save data via the `out_dir` param.

Please specify whether the data is in video or image mode through the parameter `--video_mode`, which is boolean.

There are FOUR ways to generate annotations:
1. Use `--num_images n` to generate the annotations for n images
2. Use `--all` to generate the annotations for all images
3. Use `--image_paths {p1} {p2} ...` to generate the annotations for specific images
4. Use `--annot-uuids {uuid1} {uuid2} ...` to generate for specific annotations

To specify which fields to include on each annotation, see the following list of optional parameters:

- `--detection_class` -- The class returned by the detector
- `--species` -- The species returned by the species identifier
- `--tracking_id` -- The tracking id assigned by the detector
- `--confidence` -- The strength of the bounding box detection from the detector
- `--viewpoint` -- The viewpoint
- `--CA_score` -- The CA annotation score
- `--annotations_census` -- The boolean signifying whether the annotation was kept or rejected

As of now, this step will read in the CSV format outputted by CA Classifier, before the preceeding preprocessing step which converts it to JSON and reformats data. **This will be refactored once formatting for each stage is standardized and file path is stored in annotations!**

TODO RAFACTORS:
- Revise flag inputs to match a standardized JSON output
- Allow absolute paths in --image_paths
- Add more flags
- Improve output coloring & formatting

Example Executions:
python3 visualize.py {video/image data json} {CA Output CSV} {Output Directory} --num_images 100 --CA_score --species --confidence --viewpoint
python3 visualize.py {video/image data json} {CA Output CSV} {Output Directory} --video_mode --image_paths {p1} {p2} {p3} ... --CA_score --species --confidence --viewpoint

python3 visualize.py /fs/ess/PAS2136/ggr_data/results/GGR2020_subset/image_data.json /fs/ess/PAS2136/ggr_data/results/GGR2020_subset/ca_classifier/final_output_with_softmax_and_census.csv annot_dir --all --species --viewpoint --CA_score
'''

import ast
import argparse
import os
import cv2
import json
import numbers


import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Visualize annotaions"
)
parser.add_argument(
    "metadata", type=str, help="The image/video metadata file"
)
parser.add_argument(
    "annots", type=str, help="The post CA classifier CSV file"
)
parser.add_argument(
    "out_dir", type=str, help="The directory to save annotated images to"
)
parser.add_argument(
    "--video_mode", action="store_true", help="Boolean flag identifying the format of the metadata file"
)
parser.add_argument(
    "--num_images", type=int, nargs='?', default=None, help="The number of images to display"
)
parser.add_argument(
    "--all", action="store_true", help="Displays all images"
)
parser.add_argument(
    "--image_paths", type=str, nargs='*', default=None, help="The number of images to display"
)
parser.add_argument(
    "--detection_class", action="store_true", help="Display the class returned by the detector"
)
parser.add_argument(
    "--species", action="store_true", help="Display the species returned by the species identifier"
)
parser.add_argument(
    "--tracking_id", action="store_true", help="Display the tracking id assigned by the detector"
)
parser.add_argument(
    "--confidence", action="store_true", help="Display the strength of the bounding box detection from the detector"
)
parser.add_argument(
    "--viewpoint", action="store_true", help="Display the viewpoint"
)
parser.add_argument(
    "--CA_score", action="store_true", help="Display the CA annotation score"
)
parser.add_argument(
    "--annotations_census", action="store_true", help="Display the boolean signifying whether the annotation was kept or rejected"
)

args = parser.parse_args()

# GET THE LIST OF IMAGE PATHS TO USE

with open(args.metadata, 'r') as file:
    metadata = json.load(file)

# Step 1: Get the list of all valid image uris from metadata file
if args.video_mode:
    image_metadata = []
    [image_metadata.extend(video["frame data"]) for video in metadata["videos"]]
else:
    image_metadata = metadata["images"]

uri_list = []
uri_uuid_mapping = {}
for image in image_metadata:
    uri_list.append(image["uri_original"])
    uri_uuid_mapping[image["uri_original"]] = image["uuid"]

# Step 2: Get the list of appropriate URIs to display

# Case 1: Use all images
if args.all:
    images = uri_list

# Case 2: A list was provided. Only take in valid strs (we will cross check later if they exist in annots)
elif args.image_paths is not None:
    images_input = [path for path in args.image_paths if isinstance(path,str)]
    images = list(set(images_input) & set(uri_list))

# Case 3: No list was provided. Take in at most num_images random images
elif args.num_images is not None:
    rands = np.random.choice(len(uri_list), args.num_images, replace=False)
    images = list(np.array(uri_list)[rands])

# Case 4: Invalid input
else:
    raise Exception("Invalid inputs. Must specify either num_images, image_paths, or all.")

# GET THE ANNOTATIONS CORRESPONDING TO EACH URI

images_df = pd.DataFrame(images, columns=["uri"])
images_df["image_uuid"] = images_df["uri"].map(uri_uuid_mapping)

with open(args.annots, "r") as f:
    data = json.load(f)
    annot_df = pd.DataFrame(data["annotations"])

# Inner join because we can process this via a select w/ no returns
df = pd.merge(images_df, annot_df, on="image_uuid", how="inner")

# FORMATTER
keys = []
args_dict = vars(args)
for key in args_dict:
    if key not in ["metadata", "annots", "video_mode", "num_images", "image_paths", "out_dir", "all"] and args_dict[key]:
        keys.append(key)

def format(annot):
    format_str = ""
    # Only include fields with values (and not NaN)
    for key in keys:
        if annot[key] and not (isinstance(annot[key], numbers.Number) and np.isnan(annot[key])):
            format_str += key + f" {annot[key]} "

    return format_str

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir, exist_ok=True)

# ANNOTATE ALL IMAGES (uri is unique for uuid)

for uri in images:
    select = df[df["uri"] == uri]

    # Add the annots to the image
    for _, row in select.iterrows():
        img = cv2.imread(uri)
        annot_str = format(row)
        bbox = np.array(row["bbox"]).astype(int)
        color = np.random.randint(0, 256, 3).tolist()
        # Bounding Box
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        img = cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness=3)

        # Text (commented lines are text placed ON bbox)
        (tw, th), _ = cv2.getTextSize(annot_str, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
        # img = cv2.rectangle(img, (x1, y1 - th - 2), (x1 + tw + 2, y1), color, -1)
        img = cv2.rectangle(img, (0, 0), (tw + 2, th + 2), color, -1)
        # Get the grayscale version of color s.t. we can contrast it
        gray_color = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
        if gray_color < 127.5:
            text_color = [255,255,255]
        else:
            text_color = [0,0,0]
        # img = cv2.putText(img, annot_str, (x1-1,y1+1), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 2)
        img = cv2.putText(img, annot_str, (1,th+1), cv2.FONT_HERSHEY_SIMPLEX, 3, text_color, 3)

        # Save the image on an annotation basis
        path = os.path.join(args.out_dir,row["uuid"] + ".jpg")
        cv2.imwrite(path, img)
