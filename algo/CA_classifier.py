import argparse
import ast
import os
import shutil
import warnings
import json

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.models import resnet50
from torchvision.ops import nms


def xywh_to_xyxy(bbox: list):
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.img_data = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):

        # Read image as PIL Image
        image = Image.open(self.img_data.iloc[idx]["image_path"]).convert("RGB")

        # Get the bounding box coordinates
        bbox = xywh_to_xyxy(self.img_data.iloc[idx]["bbox"])

        # Crop the image according to bbox
        image = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))

        # Flip the image if left viewpoint
        if "left" in self.img_data.iloc[idx]["predicted_viewpoint"]:
            image = F.hflip(image)

        if self.transform:
            image = self.transform(image)

        return image


class BinaryClassResNet50(nn.Module):
    def __init__(self):
        super(BinaryClassResNet50, self).__init__()
        self.resnet50 = resnet50(pretrained=True)
        for param in self.resnet50.parameters():
            param.requires_grad = False  # Freeze parameters of pre-trained model
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, 2)  # We have two classes either 0 or 1

    def forward(self, x):
        x = self.resnet50(x)
        return x


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_model(model_path, device):
    model = BinaryClassResNet50()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_data(csv_path, image_dir, transform, batch_size):
    dataset = CustomImageDataset(csv_path, image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def preprocess_viewpoint(viewpoint):
    # IF NOT A STRING (NAN)
    if not isinstance(viewpoint, str):
        return ""
    elif "front" in viewpoint and "right" in viewpoint:
        if "up" in viewpoint:
            return "upfrontright"
        else:
            return "frontright"
    elif "back" in viewpoint and "right" in viewpoint:
        if "up" in viewpoint:
            return "upbackright"
        else:
            return "backright"
    elif "front" in viewpoint and "left" in viewpoint:
        if "up" in viewpoint:
            return "upfrontleft"
        else:
            return "frontleft"
    elif "back" in viewpoint and "left" in viewpoint:
        if "up" in viewpoint:
            return "upbackleft"
        else:
            return "backleft"
    elif "right" in viewpoint:
        return "right"
    elif "left" in viewpoint:
        return "left"
    return viewpoint


def filter_dataframe(df, config):
    # Preprocess the predicted_viewpoint column
    df["predicted_viewpoint"] = df["predicted_viewpoint"].apply(preprocess_viewpoint)

    # # Filter conditions
    # bbox_condition = (
    #     df["bbox x"].notna()
    #     & df["bbox y"].notna()
    #     & df["bbox w"].notna()
    #     & df["bbox h"].notna()
    # )
    viewpoint_condition = df["predicted_viewpoint"].isin(config["viewpoints"])
    # Special condition - only applies to gt annotated data where the species was correctly identified
    if "annot species" in df.keys():
        species_condition = df["annot_species"] == config["species"]
    else:
        species_condition = True

    # Create a mask for rows to be filtered out
    filter_mask = ~(species_condition & viewpoint_condition)
    # filter_mask = ~(bbox_condition & species_condition & viewpoint_condition)

    # Split the dataframe
    filtered_out = df[filter_mask].copy().reset_index(drop=True)
    filtered_test = df[~filter_mask].copy().reset_index(drop=True)

    # Add NaN columns to filtered_out
    filtered_out["softmax_output_0"] = np.nan
    filtered_out["softmax_output_1"] = np.nan

    return filtered_test, filtered_out


def test_new(dataloader, model, device):
    model.eval()
    all_softmax_outputs = []

    with torch.no_grad():
        for X in dataloader:  #
            X = X.to(device)
            pred = model(X)
            pred_softmax = torch.softmax(pred, dim=1)
            all_softmax_outputs.extend(pred_softmax.cpu().numpy())

    all_softmax_outputs = np.array(all_softmax_outputs)
    return all_softmax_outputs


def apply_nms(df, iou_threshold):
    df = df.sort_values("softmax_output_1", ascending=False)
    boxes = np.array([xywh_to_xyxy(bbox) for bbox in df["bbox"]])
    scores = df["softmax_output_1"].values
    boxes = torch.as_tensor(boxes).float()
    scores = torch.as_tensor(scores).float()
    keep = nms(boxes, scores, iou_threshold)
    return df.iloc[keep]


def format_and_save(df, path):
    # file_name	tracking_id	confidence	detection_class	species	bbox	viewpoint	individual_id	CA_score	annotations_census
    df = df.rename(
        columns={
            "annot uuid": "uuid",
            "bbox_pred_score": "confidence",
            "category_id": "detection_class",
            "species_prediction": "species",
            "species_pred_simple": "category_id",
            "predicted_viewpoint": "viewpoint",
        }
    )
    df["individual_id"] = 0

    columns_kept = [
        "image_uuid",
        "uuid",
        "tracking_id",
        "confidence",
        "detection_class",
        "species",
        "bbox",
        "viewpoint",
        "individual_id",
        "CA_score",
        "annotations_census",
        "category_id",
        "frame_number",
        "timestamp",
        "image_path",
    ]

    df = df.drop(columns=df.columns.difference(columns_kept))

    # Save to csv
    with open(path, "w") as f:
        df.to_csv(f, index=False)


def load_annotations_from_json(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data["annotations"])


def expand_bbox_columns(df):
    df_copy = df.copy()

    # Extract bbox components into separate columns
    bbox_data = df_copy["bbox"].apply(
        lambda x: pd.Series(x, index=["bbox x", "bbox y", "bbox w", "bbox h"])
    )

    # Add the new columns to the dataframe
    df_copy = pd.concat([df_copy, bbox_data], axis=1)

    return df_copy


def main(args):
    """
    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/CA_classifier.py

    Example:
        >>> from argparse import Namespace
        >>> args = Namespace(
        ...     image_dir="test_imgs",
        ...     in_csv_path="temp/viewpoint/viewpoint_output.csv",
        ...     model_checkpoint_path="test_dataset/best_july_8_cuda_extended_aug_sample_annot_classifier.pth",
        ...     out_csv_path="temp/ca/ca_output.csv"
        ... )
        >>> main(args) # doctest: +ELLIPSIS
        Loading configuration...
        Setting up device...
        Loading and preprocessing data...
        The length of input CSV is: 40
        Setting up transformations and data loader...
        Loading model...
        Starting testing...
        Testing completed. Appending softmax outputs to CSV and starting post-processing...
        The length of softmax thresholded CSV is: 18
        The length of AR thresholded CSV is: 18
        The length of NMS thresholded CSV is: 17
        The length of NMS thresholded CSV is: 0
        The length of final concatenated CSV is: 23
        Removing Previous Instance of Experiment...
        Saving the results...
        CSV with softmax outputs and census annotations saved to: temp/ca/ca_output.csv
        All tasks completed successfully!
    """

    print("Loading configuration...")
    config = load_config("algo/CA_classifier.yaml")

    print("Setting up device...")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    print("Loading and preprocessing data...")
    df = load_annotations_from_json(args.in_csv_path)

    # Expand bbox column into separate x, y, w, h columns
    df = expand_bbox_columns(df)

    print(f"The length of input CSV is: {len(df)}")
    filtered_test, filtered_out = filter_dataframe(df, config)

    print("Setting up transformations and data loader...")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = CustomImageDataset(filtered_test, args.image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    print("Loading model...")
    with warnings.catch_warnings():  # Add this line
        warnings.filterwarnings("ignore", category=UserWarning)
        model = load_model(args.model_checkpoint_path, device)

    print("Starting testing...")
    all_softmax_outputs = test_new(dataloader, model, device)  #

    print(
        "Testing completed. Appending softmax outputs to CSV and starting post-processing..."
    )
    filtered_test["softmax_output_0"] = all_softmax_outputs[:, 0]
    filtered_test["softmax_output_1"] = all_softmax_outputs[:, 1]

    # Step 1: Filter based on threshold_CA
    above_threshold = filtered_test[
        filtered_test["softmax_output_1"] > config["threshold_CA"]
    ].reset_index(drop=True)
    below_threshold = filtered_test[
        filtered_test["softmax_output_1"] <= config["threshold_CA"]
    ].reset_index(drop=True)

    print(f"The length of softmax thresholded CSV is: {len(above_threshold)}")

    # Step 2: Filter based on log(aspect_ratio)
    above_threshold["log_AR"] = np.log(
        above_threshold["bbox w"] / above_threshold["bbox h"]
    )
    ar_filtered = above_threshold[
        (above_threshold["log_AR"] >= config["min_log_AR"])
        & (above_threshold["log_AR"] <= config["max_log_AR"])
    ].reset_index(drop=True)
    ar_filtered_out = above_threshold[
        (above_threshold["log_AR"] < config["min_log_AR"])
        | (above_threshold["log_AR"] > config["max_log_AR"])
    ].reset_index(drop=True)

    print(f"The length of AR thresholded CSV is: {len(ar_filtered)}")

    # Step 3: Apply NMS
    grouped = ar_filtered.groupby("image_path")
    all_results = []
    nms_filtered_out = []
    for name, group in grouped:
        result_df = apply_nms(group, config["NMS_threshold"])
        all_results.append(result_df)
        # Keep track of removed annotations
        removed = group[~group.index.isin(result_df.index)]
        nms_filtered_out.append(removed)
    if all_results:
        nms_filtered = pd.concat(all_results).reset_index(drop=True)
        print(f"The length of NMS thresholded CSV is: {len(nms_filtered)}")
    else:
        print("Warning: No objects passed NMS filtering")
        nms_filtered = pd.DataFrame(
            columns=ar_filtered.columns
        )  # Create empty DataFrame with same columns
    if nms_filtered_out:
        nms_filtered_out = pd.concat(nms_filtered_out).reset_index(drop=True)
    else:
        nms_filtered_out = pd.DataFrame(columns=ar_filtered.columns)
    # print(nms_filtered_out)
    print(f"The length of NMS thresholded CSV is: {len(nms_filtered)}")

    # Add annotations_census column
    nms_filtered["annotations_census"] = True
    below_threshold["annotations_census"] = False
    ar_filtered_out["annotations_census"] = False
    nms_filtered_out["annotations_census"] = False
    filtered_out["annotations_census"] = False

    # Concatenate all dataframes
    final_df = pd.concat(
        [
            nms_filtered,
            below_threshold,
            ar_filtered_out,
            nms_filtered_out,
            filtered_out,
        ],
        ignore_index=True,
    )

    # Drop specified columns
    columns_to_drop = [
        "softmax_output_0",
        "log_AR",
        "bbox x",
        "bbox y",
        "bbox w",
        "bbox h",
    ]
    final_df = final_df.drop(
        columns=[col for col in columns_to_drop if col in final_df.columns]
    )
    final_df = final_df.rename(columns={"softmax_output_1": "CA_score"})

    # Extract file_name from image path
    final_df["file_name"] = final_df["image_path"].apply(os.path.basename)

    print(f"The length of final concatenated CSV is: {len(final_df)}\n")

    # Save the updated DataFrame to a new CSV file
    cac_dir = os.path.dirname(args.out_csv_path)
    if os.path.exists(cac_dir):
        print("Removing Previous Instance of Experiment...")
        shutil.rmtree(cac_dir)

    print("Saving the results...")
    os.makedirs(cac_dir, exist_ok=True)
    format_and_save(final_df, args.out_csv_path)

    print(
        f"CSV with softmax outputs and census annotations saved to: {args.out_csv_path}"
    )
    print("All tasks completed successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run viewpoint classifier for database of animal images"
    )
    parser.add_argument(
        "image_dir", type=str, help="The directory where localized images are found"
    )
    parser.add_argument(
        "in_csv_path",
        type=str,
        help="The full path to the viewpoint classifier output csv to use as input",
    )
    parser.add_argument(
        "model_checkpoint_path", type=str, help="The full path to the model checkpoint"
    )
    parser.add_argument(
        "out_csv_path", type=str, help="The full path to the output csv file"
    )
    args = parser.parse_args()

    main(args)
