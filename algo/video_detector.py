import json
import shutil
import subprocess
import uuid

import cv2
import os
import yaml
import warnings
import argparse

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


def save_annotations_to_csv(annotations_dict, file_path):
    annotations = annotations_dict["annotations"]
    df = pd.DataFrame(annotations)
    df.to_csv(file_path, index=False)


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


# def detect_frame_sequence(frames, model, threshold):
#     # Capture the video and get name
#     sep = video_path.rfind("/")
#     vid_name = video_path[sep:].replace("/","")
#     cap = cv2.VideoCapture(video_path)

#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     num_digits = len(str(length))

#     fcount = 0
#     annotations = []
#     with tqdm(desc=f"Detecting frames from {vid_name}...",total=length) as pbar:
#         while True:
#             fcount += 1
#             # Capture frame-by-frame
#             ret, frame = cap.read()
#             if not ret:
#                 break
              
#             pbar.update(1)

#             # Save frame and get uuid info, fill with leading zeros s.t. frames appear in sorted order
#             f_name = vid_name + "_" + str(fcount).zfill(num_digits) + ".jpg"
#             f_path = os.path.join(image_dir,f_name)
#             cv2.imwrite(f_path,frame)
#             f_info = preproc.parse_imageinfo(f_path)

#             # Run YOLO detection and tracking
#             results = model.track(frame, conf=threshold, verbose=False, persist=True)

#             # Extract detections and tracking information
#             for result in results:
#                 # Check if any detection in the image is a person (class 0)
#                 if any(box.cls.item() == 0 for box in result.boxes):
#                     # Skip this entire image
#                     continue

#                 for box in result.boxes:
#                     # Process the image only if no person was detected
#                     x1 = box.xyxy[0][0].item()
#                     y1 = box.xyxy[0][1].item()
#                     x2 = box.xyxy[0][2].item()
#                     y2 = box.xyxy[0][3].item()

#                     annotations.append(
#                         {
#                             "annot uuid": str(uuid.uuid4()),
#                             "image uuid": f_info[0],
#                             "image fname": f_name,
#                             "video fname": vid_name,
#                             "frame number": fcount,
#                             "bbox x": x1,
#                             "bbox y": y1,
#                             "bbox w": x2 - x1,
#                             "bbox h": y2 - y1,
#                             "bbox pred score": box.conf.item(),
#                             "category id": int(box.cls.item()),
#                             "tracking id": int(box.id.cpu())
#                         }
#                     )

#     print(f"Finished detecting frames from {vid_name}.")
#     return annotations


def detect_videos(video_data, model, threshold):
    videos = video_data["videos"]
    annotations = []

    for vid in videos:
        vid_name = vid["video fname"]
        frames = vid["frame data"]
        fcount = 0

        for frame in tqdm(frames,desc=f"Detecting frames from {vid_name}..."):
            fcount += 1
            # Open image
            img = cv2.imread(frame["uri"])
            # Run YOLO detection and tracking
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
                            "annot uuid": str(uuid.uuid4()),
                            "image uuid": frame["uuid"],
                            "image fname": frame["original_name"],
                            "video fname": vid["video fname"],
                            "frame number": fcount,
                            "bbox x": x1,
                            "bbox y": y1,
                            "bbox w": x2 - x1,
                            "bbox h": y2 - y1,
                            "bbox pred score": box.conf.item() if box.conf is not None else -1,
                            "category id": int(box.cls.item()) if box.cls is not None else -1,
                            "tracking id": int(box.id.item()) if box.id is not None else -1,
                        }
                    )

        print(f"Finished detecting frames from {vid_name}.")
    print(annotations)
    return annotations
             
def main(args):
    config = load_config("algo/detector.yaml")

    # Setting up Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.exp_dir)
    annots = Path(args.annot_dir)
    filtered_annot_csv_path = os.path.join(annots, args.annots_filtered_csv_filename) + ".csv"

    with open(args.video_data, 'r') as file:
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
    print(f"Running detection on all videos...")
    annotations = detect_videos(video_data, detector, threshold)

    print(f"Saving annotations to {filtered_annot_csv_path}...")
    save_annotations_to_csv({"annotations": annotations},filtered_annot_csv_path)

    print(f"Done!")


if __name__ == '__main__':
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
