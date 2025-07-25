import argparse
import json
import os
import re
import shutil
import subprocess
import uuid
import warnings

import cv2
import torch
import ultralytics
import yaml
from tqdm import tqdm

from VAREID.libraries.utils import path_from_file

ultralytics.checks()
import urllib.request
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from VAREID.libraries.io.format_funcs import clone_from_github, load_config, load_json, save_json, split_dataframe

warnings.filterwarnings("ignore")


def download_model(yolo_model, yolo_url, model_dir):
    if yolo_url:
        file_name = os.path.join(model_dir, yolo_model) + ".pt"
        urllib.request.urlretrieve(yolo_url, file_name)
        print(f"Successfully Downloaded {yolo_model} to {file_name}")
    else:
        print(f"URL for version {yolo_model} is not available ... ")


def select_model(yolo_model, config, model_dir):
    url_key = f"{yolo_model}_url"
    yolo_url = config.get(url_key)
    download_model(yolo_model, yolo_url, model_dir)
    model = YOLO(os.path.join(model_dir, yolo_model + ".pt"))
    return model


def detect_videos(video_data, model, threshold, sz):
    videos = video_data["videos"]
    annotations = []

    for vid in videos:
        vid_name = vid["video fname"]
        frames = vid["frame data"]
        fcount = 0

        for frame in tqdm(frames, desc=f"Detecting frames from {vid_name}..."):
            fcount += 1
            # Open image
            img = cv2.imread(frame["uri"])
            # Run YOLO detection and tracking
            if fcount == 1:
                results = model.track(img, conf=threshold, verbose=False, persist=False, imgsz=sz)
            else:
                results = model.track(img, conf=threshold, verbose=False, persist=True, imgsz=sz)

            # Extract detections and tracking information
            for result in results:
                # Check if any detection in the image is a person (class 0)
                if any(box.cls.item() == 0 for box in result.boxes):
                    # Skip this entire image
                    continue

                for box in result.boxes:
                    # Process the image only if no person was detected
                    x1 = box.xyxy[0][0].item()
                    y1 = box.xyxy[0][1].item()
                    x2 = box.xyxy[0][2].item()
                    y2 = box.xyxy[0][3].item()

                    annotations.append(
                        {
                            "uuid": str(uuid.uuid4()),
                            "image_uuid": frame["uuid"],
                            "image_path": frame["uri"],
                            "video_path": vid["video path"],
                            "frame_number": fcount,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "confidence": (
                                box.conf.item() if box.conf is not None else -1
                            ),
                            "detection_class": (
                                int(box.cls.item()) if box.cls is not None else -1
                            ),
                            "tracking_id": (
                                int(box.id.item()) if box.id is not None else -1
                            ),
                            "timestamp": frame["time_posix"],
                        }
                    )

        print(f"Finished detecting frames from {vid_name}.")
    return annotations


def postprocess_tracking_ids(annots):
    """
    Ensures that tracking ids for separate videos do not overlap in the same range of numbers
    by remapping tracking ids to unused integer values.
    """

    # Defines which video paths use which tracking id. tracking id -> video path
    used_keys = {}
    # Defines mappings to follow. (video path, tracking id) -> new tracking id
    mappings = {}
    # The smallest unused tracking id
    next_unused_id = 1

    for index, annot in enumerate(annots):
        tid = annot["tracking_id"]
        path = annot["video_path"]

        # Check if the key is used by a different image
        if tid in used_keys.keys() and used_keys[tid] != path:
            # If it is, check if a mapping exists yet
            mapping_key = (path, tid)
            if mapping_key in mappings.keys():
                tid = mappings[mapping_key]
            # If it doesn't create the mapping
            else:
                mappings[mapping_key] = next_unused_id
                tid = next_unused_id

        # Mark the new key if needed
        if tid not in used_keys.keys():
            used_keys[tid] = path
            # Find the next unused id not in used_keys
            while next_unused_id in used_keys.keys():
                next_unused_id += 1

        # Assign the tid
        annots[index]["tracking_id"] = tid


def main(args):
    config = load_config(path_from_file(__file__, "detector_config.yaml"))
    
    dt_dir = Path(args.dt_dir)

    video_data = load_json(args.video_data)

    os.makedirs(dt_dir, exist_ok=True)

    yolo_dir = os.path.join(dt_dir, config["yolo_dir"])
    github_v10_url = config["github_v10_url"]

    clone_from_github(yolo_dir, github_v10_url)

    model_dir = os.path.join(yolo_dir, config["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    yolo_model = args.model_version
    detector = select_model(yolo_model, config, model_dir)

    threshold = config["confidence_threshold"]
    sz = config["img_size_vid"]

    # Detect and track objects over all videos
    print("Running detection on all videos...")
    annotations = detect_videos(video_data, detector, threshold, sz)

    print("Postprocessing tracking ids to avoid collisions...")
    postprocess_tracking_ids(annotations)

    df = pd.DataFrame(annotations)
    annotations = split_dataframe(df)

    print(f"Saving annotations to {args.out_json_path}...")
    filtered_annot_json_path = os.path.join(dt_dir, args.out_json_path)
    save_json(annotations, filtered_annot_json_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and track bounding boxes for database of animal videos"
    )
    parser.add_argument(
        "video_data", type=str, help="The video metadata file"
    )
    parser.add_argument(
        "dt_dir", type=str, help="The directory to export models and annots to"
    )
    parser.add_argument(
        "model_version", type=str, help="The yolo model version to use"
    )
    parser.add_argument(
        "out_json_path", type=str, help="The name of the output annotations json file"
    )

    args = parser.parse_args()

    main(args)
