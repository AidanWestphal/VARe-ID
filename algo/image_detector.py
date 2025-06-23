import argparse
import json
import os
import shutil
import subprocess
import uuid
import warnings

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


def load_annotations_from_csv(csv_file_path):
    if not os.path.exists(csv_file_path):
        return None
    df = pd.read_csv(csv_file_path)
    return df


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
    print(f"Model device: {next(model.parameters()).device}")
    return model


def detect_images(image_data, model, threshold, sz):

    images = image_data["images"]
    tracking_id = 1

    # ANNOTION LIST FOR OUTPUT
    annotations = []

    for image in tqdm(images, desc=f"Detecting images..."):
        # Detect from the image
        results = model(image["uri_original"], conf=threshold, imgsz=sz, verbose=False)

        for result in results:
            # Check if any detection in the image is a person (class 0)
            if any(box.cls.item() == 0 for box in result.boxes):
                # Skip this entire image
                continue

            # Process the image only if no person was detected
            for box in result.boxes:
                x1 = box.xyxy[0][0].item()
                y1 = box.xyxy[0][1].item()
                x2 = box.xyxy[0][2].item()
                y2 = box.xyxy[0][3].item()

                # ADD THE ANNOTATION TO THE LIST OF ANNOTS
                annotations.append(
                    {
                        "uuid": str(uuid.uuid4()),
                        "image_uuid": image["uuid"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "confidence": box.conf.item(),
                        "detection_class": int(box.cls.item()),
                        "tracking_id": tracking_id,
                        "timestamp": image["time_posix"],
                        "image_path": image["uri_original"],
                    }
                )

                # For images, assign unique tracking ids to every image
                tracking_id += 1
    
    # OBTAIN THE SET OF UNIQUE IMAGES TO PUT IN ANNOTATIONS
    df = pd.DataFrame(annotations)
    df_images = (
        df[["image_uuid", "image_path", "timestamp"]].drop_duplicates(keep="first").reset_index(drop=True)
    )
    df_images = df_images.rename(columns={"image_uuid": "uuid"})

    annotations_dict = {"images": df_images.to_dict(orient="records"), "annotations": annotations}
    print(", done.")

    return annotations_dict


def save_annotations_to_json(annotations_dict, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(annotations_dict, json_file, indent=4)


def load_annotations_from_json(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data["annotations"])


def calculate_iou(box1: list, box2: list):
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
    pred_df = predicted_df

    filtered_annotations = []

    for index, row in pred_df.iterrows():

        x0, y0, w, h = row["bbox"]
        pred_bbox = [x0, y0, x0 + w, y0 + h]

        image_uuid = row["image_uuid"]
        image_fname = row["image_fname"]

        image_df = original_df[original_df["image_uuid"] == row["image_uuid"]]

        print(f"Filtering annotations: ({index + 1}/{len(pred_df)})", end="")
        if index < len(pred_df) - 1:
            print("\r", end="")

        keep = False

        for _, org_row in image_df.iterrows():
            org_bbox_x0 = org_row["bbox"][0]
            org_bbox_y0 = org_row["bbox"][1]
            org_bbox_x1 = org_row["bbox"][2] + org_bbox_x0
            org_bbox_y1 = org_row["bbox"][3] + org_bbox_y0

            org_bbox = [org_bbox_x0, org_bbox_y0, org_bbox_x1, org_bbox_y1]

            if calculate_iou(pred_bbox, org_bbox) >= iou_thresh:
                keep = True
                species = org_row["annot_species"]
                ca = org_row["annot_census"]
                viewpoint = org_row["viewpoint"]
                break

        if keep:
            annotation = row.to_dict()
            annotation["annot_species"] = species
            annotation["bbox_gt"] = [int(coord) for coord in org_bbox]
            annotation["bbox_pred"] = [int(coord) for coord in pred_bbox]
            annotation["bbox_pred"] = [x0, y0, w, h]
            annotation["viewpoint"] = viewpoint
            annotation["annot_census"] = ca
            annotation["image_uuid"] = image_uuid
            annotation["image_fname"] = image_fname
            filtered_annotations.append(annotation)

    filtered_annotations = [
        {
            "annot_uuid": annotation["annot_uuid"],
            "image_uuid": annotation["image_uuid"],
            "bbox_pred_score": annotation["bbox_pred_score"],
            "category_id": annotation["category_id"],
            "annot_species": (
                "neither"
                if annotation["annot_species"] not in ["zebra_grevys", "zebra_plains"]
                else annotation["annot_species"]
            ),
            "annot_census": annotation["annot_census"],
            "bbox_gt": annotation["bbox_gt"],
            "bbox_pred": annotation["bbox_pred"],
            "bbox_x": annotation["bbox_pred"],
            "viewpoint_gt": annotation["viewpoint"],
            "image_fname": annotation["image_fname"],
            "timestamp": annotation["timestamp"],
        }
        for annotation in filtered_annotations
    ]

    print(
        f", done.\nFiltered annotations: ({len(filtered_annotations)}/{len(pred_df)} total annotations)"
    )
    filtered_annotations_dict = {"annotations": filtered_annotations}
    return filtered_annotations_dict


def main(args):
    """
    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/detector.py

    Example:
        >>> from argparse import Namespace
        >>> args = Namespace(image_dir="test_imgs", annot_dir="temp/annots",
        ...                  exp_dir="temp/models", model_version="yolov10l",
        ...                  annots_csv_filename="annots", annots_filtered_csv_filename="filtered_annots",
        ...                  original_csv_path="path/to/csv")
        >>> main(args) # doctest: +ELLIPSIS
            Cloning repository https://github.com/....git into temp/models/Yolo_v10...
            Successfully Downloaded yolov10l to temp/models/Yolo_v10/weights/yolov10l.pt
            Running detection... done.
            , done.
            Saving annotations to JSON: temp/annots/annots.json
            Saving annotations to CSV: temp/annots/annots.csv
            Loading ground truth annotations...
            No ground truth annotations detected. Skipped filtering.
            Saving non-filtered annotations to JSON: temp/annots/filtered_annots.json
            Saving annotations to CSV: temp/annots/filtered_annots.csv
    """
    # Loading Configuration File ...
    config = load_config("algo/detector.yaml")

    # Setting up Device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    annotations_csv_fullpath = args.original_csv_path
    exp_dir = Path(args.exp_dir)
    annots = Path(args.annot_dir)

    with open(args.image_data, "r") as file:
        image_data = json.load(file)

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

    predictions = detect_images(image_data, detector, config["confidence_threshold"], config["img_size"])

    pred_json_name = args.annots_csv_path

    annot_json_path = os.path.join(annots, pred_json_name)
    print("Saving annotations to JSON:", annot_json_path)
    save_annotations_to_json(predictions, annot_json_path)

    print("Loading ground truth annotations...")
    base_df = load_annotations_from_csv(annotations_csv_fullpath)

    if base_df is not None:
        pred_df = load_annotations_from_json(annot_json_path)

        filtered_annotations = filtration(pred_df, base_df)

        filtered_pred_json_name = args.annots_filtered_csv_path

        filtered_annot_json_path = os.path.join(annots, filtered_pred_json_name)
        save_annotations_to_json(filtered_annotations, filtered_annot_json_path)
    else:
        print("No ground truth annotations detected. Skipped filtering.")
        non_filtered_pred_json_name = args.annots_filtered_csv_path

        non_filtered_annot_json_path = os.path.join(annots, non_filtered_pred_json_name)
        print("Saving non-filtered annotations to JSON:", non_filtered_annot_json_path)
        save_annotations_to_json(predictions, non_filtered_annot_json_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Detect bounding boxes for database of animal images"
    )
    parser.add_argument("image_data", type=str, help="The image metadata file")
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
        "annots_csv_path",
        type=str,
        help="The name of the output annotations csv file",
    )
    parser.add_argument(
        "annots_filtered_csv_path",
        type=str,
        help="The name of the output filtered annotations csv file",
    )

    args = parser.parse_args()

    main(args)
