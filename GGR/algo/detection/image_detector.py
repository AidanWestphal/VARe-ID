import argparse
import os
import shutil
import subprocess
import sys
import uuid
import warnings
import ultralytics
import urllib.request

from tqdm import tqdm
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

from GGR.util.io.format_funcs import clone_from_github, load_config, load_json, load_dataframe, save_json, split_dataframe
from GGR.util.utils import path_from_file

ultralytics.checks()
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
            # if len(result.boxes) == 0:
            #     print(image)
            #     img = Image.open(image["uri_original"])

            #     # ADD THE ANNOTATION TO THE LIST OF ANNOTS
            #     annotations.append(
            #         {
            #             "annot_uuid": str(uuid.uuid4()),
            #             "image_uuid": image["uuid"],
            #             "bbox": [0, 0, img.size[0], img.size[1]],
            #             "confidence": 0,
            #             "detection_class": 0,
            #             "tracking_id": tracking_id,
            #             "timestamp": image["time_posix"],
            #             "image_path": image["uri_original"],
            #         }
            #     )
    
    print(", done.")
    return annotations


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


def compare(predicted_df, original_df):
    pred_df = predicted_df

    compared_annotations = []

    for index, row in pred_df.iterrows():

        x0, y0, w, h = row["bbox"]
        pred_bbox = [x0, y0, x0 + w, y0 + h]

        image_df = original_df[original_df["image_uuid"] == row["image_uuid"]]

        print(f"Comparing annotations: ({index + 1}/{len(pred_df)})", end="")
        if index < len(pred_df) - 1:
            print("\r", end="")

        max_iou = -1
        for _, org_row in image_df.iterrows():
            org_bbox_x0 = org_row["bbox"][0]
            org_bbox_y0 = org_row["bbox"][1]
            org_bbox_x1 = org_row["bbox"][2] + org_bbox_x0
            org_bbox_y1 = org_row["bbox"][3] + org_bbox_y0

            org_bbox = [org_bbox_x0, org_bbox_y0, org_bbox_x1, org_bbox_y1]
            iou = calculate_iou(pred_bbox, org_bbox)
            if iou >= max_iou:
                max_iou = iou
        
        annotation = row.to_dict()
        annotation["gt_iou"] = max_iou
        compared_annotations.append(annotation)

    print(
        f", done.\nCompared annotations: ({len(compared_annotations)}/{len(pred_df)} total annotations)"
    )
    compared_annotations_dict = {"annotations": compared_annotations}
    return compared_annotations_dict


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
    config = load_config(path_from_file(__file__, "detector_config.yaml"))

    dt_dir = Path(args.dt_dir)
    annots = Path(args.annot_dir)

    image_data = load_json(args.image_data)

    os.makedirs(dt_dir, exist_ok=True)
    shutil.rmtree(annots, ignore_errors=True)
    os.makedirs(annots, exist_ok=True)

    yolo_dir = os.path.join(dt_dir, config["yolo_dir"])
    github_v10_url = config["github_v10_url"]

    clone_from_github(yolo_dir, github_v10_url)

    model_dir = os.path.join(yolo_dir, config["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    yolo_model = args.model_version
    detector = select_model(yolo_model, config, model_dir)

    predictions = detect_images(image_data, detector, config["confidence_threshold"], config["img_size"])

    print("Splitting annotations...")
    df = pd.DataFrame(predictions)
    predictions = split_dataframe(df)

    # SAVE ALL DATA 
    non_filtered_pred_json_name = args.out_json_path

    non_filtered_annot_json_path = os.path.join(annots, non_filtered_pred_json_name)
    print("Saving non-filtered annotations to JSON:", non_filtered_annot_json_path)
    save_json(predictions, non_filtered_annot_json_path)

    # FILTER PENDING PRESENCE OF GT DATA
    if args.gt_path and args.gt_filtered_annots:
        print("Ground truth data and save path detected.")
        
        print("Loading ground truth annotations...")
        base_df = load_dataframe(args.gt_path)

        pred_df = df
        base_df = pd.DataFrame(base_df)

        compared_annotations = compare(pred_df, base_df)

        compared_dict = predictions.copy()
        compared_dict["annotations"] = compared_annotations["annotations"]

        compared_pred_json_name = args.gt_filtered_annots

        print("Saving filtered ground truth annotations...")
        compared_annot_json_path = os.path.join(annots, compared_pred_json_name)
        save_json(compared_dict, compared_annot_json_path)

    elif not args.gt_path and not args.gt_filtered_annots:
        print("No ground truth annotations detected. Skipped filtering.")
    else:
        print("Either a ground truth file or a path to save filtered annotations was provided, but not the other.", file=sys.stderr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect bounding boxes for database of animal images")
    parser.add_argument("image_data", type=str, help="The image metadata file")
    parser.add_argument("annot_dir", type=str, help="The directory to export annotations to")
    parser.add_argument("dt_dir", type=str, help="The directory to export models and predictions to")
    parser.add_argument("model_version", type=str, help="The yolo model version to use")
    parser.add_argument("out_json_path", type=str, help="The name of the output annotations json file")
    parser.add_argument("--gt_path", type=str, default=None, help="The full path to the ground truth file.")
    parser.add_argument("--gt_filtered_annots", type=str, default=None, help="The name of the output annots filted by gt data.")

    args = parser.parse_args()

    main(args)
