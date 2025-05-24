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

ultralytics.checks()
import urllib.request
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

warnings.filterwarnings("ignore")


def load_config(config_file_path):
    with open(config_file_path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file


def save_annotations_to_json(annotations_dict, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(annotations_dict, json_file, indent=4)


def clone_yolo_from_github(yolo_dir, repo_url):
    if not os.path.exists(yolo_dir) or not os.listdir(yolo_dir):
        print(f"Cloning repository {repo_url} into {yolo_dir}...")
        subprocess.run(["git", "clone", repo_url, yolo_dir])
    else:
        print(f"Directory {yolo_dir} exists and is not empty.")


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


def detect_videos(video_data, model, threshold):
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
                results = model.track(img, conf=threshold, verbose=False, persist=False)
            else:
                results = model.track(img, conf=threshold, verbose=False, persist=True)

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
                            "annot_uuid": str(uuid.uuid4()),
                            "image_uuid": frame["uuid"],
                            "image_path": frame["uri"],
                            "video_path": vid["video path"],
                            "frame_number": fcount,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "bbox_pred_score": (
                                box.conf.item() if box.conf is not None else -1
                            ),
                            "category_id": (
                                int(box.cls.item()) if box.cls is not None else -1
                            ),
                            "tracking_id": (
                                int(box.id.item()) if box.id is not None else -1
                            ),
                        }
                    )

        print(f"Finished detecting frames from {vid_name}.")
    return annotations


def parse_srt(srt_path):
    """
    Parse an SRT file and return a dict mapping:
        srt_dict[SrtCnt] = "timestamp string"
    For example, lines in the SRT might look like:

        SrtCnt : 1, DiffTime : 33ms
        2023-01-19 10:56:36,107,334

    We'll look for `SrtCnt : X` and store the next line
    (assuming it’s the date/time) as srt_dict[X].
    """
    srt_dict = {}
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for something like "SrtCnt : 123"
        match = re.search(r"SrtCnt\s*:\s*(\d+)", line)
        if match:
            srt_cnt = int(match.group(1))  # This is 1-based
            # The very next line should have the date/time
            if i + 1 < len(lines):
                possible_time = lines[i + 1].strip()
                # If it looks like a datetime, store it
                if re.search(
                    r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3},\d+", possible_time
                ):
                    srt_dict[srt_cnt] = possible_time
            i += 2
        else:
            i += 1
    return srt_dict


def add_timestamps(video_data, annots, desired_fps):
    """
    Reads input_csv with columns like: frame_name, bounding_box, etc.
    We assume frame_name is something like WD_0087_10.jpg,
    where "10" is the "extracted frame index".

    The original frame index = extracted_idx * frame_interval.
    SRT is 1-based, so we use original_idx + 1 to look up the timestamp
    from srt_dict.

    Writes a new CSV with an extra 'timestamp' column appended.
    """

    srt_table = {}
    fps_table = {}

    for index, annot in enumerate(annots):
        video_path = annot["video_path"]

        # If the associated srt was not cached, find and cache it
        if video_path not in srt_table.keys():
            # Find the matching video in video_data by the video's path. It's guaranteed that there is only one match
            data = [
                video
                for video in video_data["videos"]
                if video["video path"] == video_path
            ][0]
            srt_table[video_path] = parse_srt(data["srt path"])
            fps_table[video_path] = data["fps"]

        # Reference the srt file and assign the timestamp
        srt = srt_table[video_path]
        frame_interval = round(fps_table[video_path] / desired_fps)
        # Undo 1-indexed frames, scale by frame interval, and redo 1-index
        original_frame_number = (annot["frame_number"] - 1) * frame_interval + 1
        timestamp = srt[original_frame_number]

        # Assign the timestamp to the annotation
        annots[index]["timestamp"] = timestamp


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
    config = load_config("algo/detector.yaml")

    # Setting up Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.exp_dir)
    annots = Path(args.annot_dir)
    filtered_annot_csv_path = (
        os.path.join(annots, args.annots_filtered_csv_filename) + ".json"
    )

    with open(args.video_data, "r") as file:
        video_data = json.load(file)

    os.makedirs(exp_dir, exist_ok=True)
    shutil.rmtree(annots, ignore_errors=True)
    os.makedirs(annots, exist_ok=True)

    yolo_dir = os.path.join(exp_dir, config["yolo_dir"])
    github_v10_url = config["github_v10_url"]
    shutil.rmtree(yolo_dir, ignore_errors=True)

    clone_yolo_from_github(yolo_dir, github_v10_url)

    model_dir = os.path.join(yolo_dir, config["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    yolo_model = args.model_version
    detector = select_model(yolo_model, config, model_dir)

    threshold = config["confidence_threshold"]

    # Detect and track objects over all videos
    print("Running detection on all videos...")
    annotations = detect_videos(video_data, detector, threshold)

    print("Postprocessing tracking ids to avoid collisions...")
    postprocess_tracking_ids(annotations)

    print("Writing timestamp data from SRT files...")
    add_timestamps(video_data, annotations, config["video_fps"])

    print("Saving annotations to {filtered_annot_csv_path}...")
    save_annotations_to_json({"annotations": annotations}, filtered_annot_csv_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and track bounding boxes for database of animal videos"
    )
    parser.add_argument(
        "video_data", type=str, help="The json file with frame information stored"
    )
    parser.add_argument(
        "annot_dir", type=str, help="The directory to export annotations to"
    )
    parser.add_argument(
        "exp_dir", type=str, help="The directory to export models and predictions to"
    )
    parser.add_argument("model_version", type=str, help="The yolo model version to use")
    parser.add_argument(
        "annots_csv_filename",
        type=str,
        help="The name of the output annotations csv file",
    )
    parser.add_argument(
        "annots_filtered_csv_filename",
        type=str,
        help="The name of the output filtered annotations csv file",
    )

    args = parser.parse_args()

    main(args)
