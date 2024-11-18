import os
import ast
import cv2
import json
import yaml
import uuid
import torch
import shutil
import argparse
import warnings
import subprocess
import numpy as np
import ultralytics

ultralytics.checks()
import pandas as pd
import urllib.request
from pathlib import Path
from ultralytics import YOLO

warnings.filterwarnings("ignore")


def load_config(config_file_path):
    with open(config_file_path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file


def load_annotations_from_csv(csv_file_path):
    if not os.path.exists(csv_file_path):
        return None
    df = pd.read_csv(csv_file_path)
    return df


def clone_yolo_from_github(directory, repo_url):
    if not os.path.exists(yolo_dir) or not os.listdir(yolo_dir):
        print(f"Cloning repository {repo_url} into {directory}...")
        subprocess.run(["git", "clone", repo_url, directory])
    else:
        print(f"Directory {directory} exists and is not empty.")


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


def yolo_predictions(results, images):

    annotations = []
    for i, result in enumerate(results):

        print(f"Formatting annotations for images: ({i + 1}/{len(results)})", end="")
        if i < len(results) - 1:
            print("\r", end="")

        image_filename = os.path.basename(result.path)
        image_uuid, _ = os.path.splitext(image_filename)

        bboxes = result.boxes
        for box in bboxes:

            class_label = box.cls.item()
            # Skip if the detected class is "person" (class label 0)
            if class_label == 0:
                continue

            annot_uuid = str(uuid.uuid4())
            coordinates = box.xyxy
            conf_scores = box.conf.item()

            x1 = coordinates[0][0].item()
            y1 = coordinates[0][1].item()
            x2 = coordinates[0][2].item()
            y2 = coordinates[0][3].item()

            bbox_values = [x1, y1, x2, y2]
            annotation = {
                "uuid": annot_uuid,
                "image uuid": image_uuid,
                "bbox": bbox_values,
                "bbox pred score": conf_scores,
                "category id": int(class_label),
            }
            annotations.append(annotation)

    annotations_dict = {"annotations": annotations}
    print(", done.")

    return annotations_dict


def save_annotations_to_json(annotations_dict, file_path):
    with open(file_path, "w") as json_file:
        json.dump(annotations_dict, json_file, indent=4)


def save_annotations_to_csv(annotations_dict, file_path):
    annotations = annotations_dict["annotations"]
    df = pd.DataFrame(annotations)
    df.to_csv(file_path, index=False)


def calculate_iou(box1, box2):

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou


def filtration(predicted_df, original_df, iou_thresh=0.50):

    df = original_df
    pred_df = predicted_df

    filtered_annotations = []

    for index, row in pred_df.iterrows():

        x0, y0, x1, y1 = ast.literal_eval(row["bbox"])
        pred_bbox = [x0, y0, x1, y1]

        image_uuid = row["image uuid"]
        image_df = df[df["image uuid"] == image_uuid]

        print(f"Filtering annotations: ({index + 1}/{len(pred_df)})", end="")
        if index < len(pred_df) - 1:
            print("\r", end="")

        keep = False

        for org_index, org_row in image_df.iterrows():
            # org_bbox_x0 = org_row["bbox x"]
            # org_bbox_y0 = org_row["bbox y"]
            # org_bbox_x1 = org_row["bbox w"] + org_bbox_x0
            # org_bbox_y1 = org_row["bbox h"] + org_bbox_y0

            # Parse bbox as a list of four elements from original_df
            org_bbox = ast.literal_eval(org_row["bbox"])

            # Ensure org_bbox is a list of four values
            if isinstance(org_bbox, list) and len(org_bbox) == 4:
                org_bbox_x0, org_bbox_y0, org_bbox_x1, org_bbox_y1 = org_bbox
            else:
                raise ValueError(
                    f"Expected bbox to be a list of 4 elements, got {org_bbox}"
                )

            org_bbox = [org_bbox_x0, org_bbox_y0, org_bbox_x1, org_bbox_y1]

            iou = calculate_iou(pred_bbox, org_bbox)
            if iou >= iou_thresh:
                keep = True
                species = org_row["annot species"]
                ca = org_row["annot census"]
                image_fname = org_row["image fname"]
                break

        if keep:
            annotation = row.to_dict()
            annotation["annot species"] = species
            annotation["bbox gt"] = [int(coord) for coord in org_bbox]
            annotation["bbox pred"] = [int(coord) for coord in pred_bbox]
            annotation["bbox pred x"] = x0
            annotation["bbox pred y"] = y0
            annotation["bbox pred w"] = x1 - x0
            annotation["bbox pred h"] = y1 - y0
            annotation["annot census"] = ca
            annotation["image uuid"] = image_uuid
            annotation["image fname"] = image_fname
            filtered_annotations.append(annotation)

    filtered_annotations = [
        {
            "uuid": annotation["uuid"],
            "image uuid": annotation["image uuid"],
            "bbox pred score": annotation["bbox pred score"],
            "category id": annotation["category id"],
            "annot species": (
                "neither"
                if annotation["annot species"] not in ["zebra_grevys", "zebra_plains"]
                else annotation["annot species"]
            ),
            "annot census": annotation["annot census"],
            "bbox gt": annotation["bbox gt"],
            "bbox pred": annotation["bbox pred"],
            "bbox x": annotation["bbox pred x"],
            "bbox y": annotation["bbox pred y"],
            "bbox w": annotation["bbox pred w"],
            "bbox h": annotation["bbox pred h"],
            "image fname": annotation["image fname"],
        }
        for annotation in filtered_annotations
    ]

    print(
        f", done.\nFiltered annotations: ({len(filtered_annotations)}/{len(pred_df)} total annotations)"
    )
    filtered_annotations_dict = {"annotations": filtered_annotations}
    return filtered_annotations_dict


if __name__ == "__main__":

    # Loading Configuration File ...
    config = load_config("algo/detector.yaml")

    # Setting up Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(
        description="Detect bounding boxes for database of animal images"
    )
    parser.add_argument(
        "image_dir", type=str, help="The directory where localized images are found"
    )
    parser.add_argument(
        "annot_dir", type=str, help="The directory to export annotations to"
    )
    parser.add_argument(
        "exp_dir", type=str, help="The directory to export models and predictions to"
    )
    parser.add_argument(
        "original_csv_path", type=str, help="The full path to the ground truth csv"
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
    annotations_csv_fullpath = args.original_csv_path
    exp_dir = Path(args.exp_dir)
    images = Path(args.image_dir)
    annots = Path(args.annot_dir)
    im_list = os.listdir(images)
    im_path = [os.path.join(images, img) for img in im_list]

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
    print("Running detection...", end="")
    results = detector(im_path, conf=threshold, verbose=False)
    print(", done.")
    predictions = yolo_predictions(results, im_path)

    pred_json_name = args.annots_csv_filename + ".json"
    pred_csv_name = args.annots_csv_filename + ".csv"

    annot_json_path = os.path.join(annots, pred_json_name)
    save_annotations_to_json(predictions, annot_json_path)

    annot_csv_path = os.path.join(annots, pred_csv_name)
    save_annotations_to_csv(predictions, annot_csv_path)

    base_df = load_annotations_from_csv(annotations_csv_fullpath)

    if base_df:
        pred_df = load_annotations_from_csv(annot_csv_path)

        filtered_annotations = filtration(pred_df, base_df)

        filtered_pred_json_name = args.annots_filtered_csv_filename + ".json"
        filtered_pred_csv_name = args.annots_filtered_csv_filename + ".csv"

        filtered_annot_json_path = os.path.join(annots, filtered_pred_json_name)
        save_annotations_to_json(filtered_annotations, filtered_annot_json_path)

        filtered_annot_csv_path = os.path.join(annots, filtered_pred_csv_name)
        save_annotations_to_csv(filtered_annotations, filtered_annot_csv_path)
    else:
        print("No ground truth annotations detected. Skipped filtering.")
