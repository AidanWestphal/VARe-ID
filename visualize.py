'''
## Annotation Visualization Tool

The following tool helps visualize annotations generated from the pipeline by overlaying them onto images.

Import data under the `annots` parameter. This should be a path to the post CA classifier annotation file. Save data via the `out_dir` param.

Please specify whether the data is in video or image mode through the parameter `video_mode`, which is boolean.

If you have a specific set of images you want displayed, you can specify the optinal field `image_paths`, which is a list of relative paths (from GGR) to each desired image. If not specified, the images are generated randomly. You can also specify the number of images to generate visualizations of, through the parameter `num_images`. Including a value for `image_paths` renders `num_images` redundant. It will generate annotations for all images in the provided list.

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
python3 visualization/visualize.py {video/image data json} {CA Output CSV} {True/False} {Output Directory} --num_images 100 --CA_score --species --confidence --viewpoint
python3 visualization/visualize.py {video/image data json} {CA Output CSV} {True/False} {Output Directory} --image_paths {p1} {p2} {p3} ... --CA_score --species --confidence --viewpoint
'''

import ast
import argparse
import os
import cv2
import json

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
    "video_mode", type=bool, help="Boolean flag identifying the format of the metadata file"
)
parser.add_argument(
    "out_dir", type=str, help="The directory to save annotated images to"
)
parser.add_argument(
    "--num_images", type=int, nargs='?', default=None, help="The number of images to display"
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

# Case 1: A list was provided. Only take in valid strs (we will cross check later if they exist in annots)
if args.image_paths is not None:
    images_input = [path for path in args.image_paths if isinstance(path,str)]
    images = list(set(images_input) & set(uri_list))

# Case 2: No list was provided. Take in at most num_images random images
elif args.num_images is not None:
    rands = np.random.choice(len(uri_list), args.num_images, replace=False)
    images = list(np.array(uri_list)[rands])

# Case 3: Invalid input
else:
    raise Exception("Invalid inputs. Must specify either num_images or image_paths.")

# GET THE ANNOTATIONS CORRESPONDING TO EACH URI

images_df = pd.DataFrame(images, columns=["uri"])
images_df["image_uuid"] = images_df["uri"].map(uri_uuid_mapping)

annot_df = pd.read_csv(args.annots)

# Inner join because we can process this via a select w/ no returns
df = pd.merge(images_df, annot_df, on="image_uuid", how="inner")

# BUILD THE FORMATTER
format_str = ""
keys = []
args_dict = vars(args)
for key in args_dict:
    if key not in ["metadata", "annots", "video_mode", "num_images", "image_paths", "out_dir"] and args_dict[key]:
        format_str += key + " {} "
        keys.append(key)

def format(annot):
    values = [annot[key] for key in keys]
    return format_str.format(*values)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir, exist_ok=True)

# ANNOTATE ALL IMAGES (uri is unique for uuid)

for uri in images:
    select = df[df["uri"] == uri]
    img = cv2.imread(uri)

    # Add the annots to the image
    for _, row in select.iterrows():
        annot_str = format(row)
        bbox = np.array(ast.literal_eval(row["bbox"])).astype(int)
        color = np.random.randint(0, 256, 3).tolist()
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, thickness=3)
        img = cv2.putText(img, annot_str, ( bbox[0], bbox[1] + bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Save the image
    path = os.path.join(args.out_dir,uri_uuid_mapping[uri] + ".jpg")
    cv2.imwrite(path, img)
