from ultralytics import YOLO

import argparse
import ast
import os
import shutil
import warnings
import json
import pickle
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.ops import nms

from VAREID.libraries.io.checkpoint import DataLoaderCheckpointManager
from VAREID.libraries.io.format_funcs import load_config, load_json, save_json, split_dataframe, join_dataframe
from VAREID.libraries.utils import path_from_file

def xywh_to_xyxy(bbox: list):
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]

def xyxy_to_xywh(bbox: list):
    x1, y1, x2, y2 = bbox
    x = x1
    y = y1
    w = x2-x1
    h = y2-y1
    return [x, y, w, h]

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.img_data = dataframe
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
        if "left" in self.img_data.iloc[idx]["viewpoint"]:
            image = F.hflip(image)

        if self.transform:
            image = self.transform(image)

        return image
    
def load_model(model_path, device):
    model = YOLO(model_path)
    model.to(device)
    model.eval()
    return model

def expand_bbox_columns(df):
    # Extract bbox components into separate columns
    bbox_data = df["bbox"].apply(
        lambda x: pd.Series(x, index=["bbox x", "bbox y", "bbox w", "bbox h"])
    )

    # Add the new columns to the dataframe
    df = pd.concat([df, bbox_data], axis=1)
    return df

def get_new_bbox(model, image):
    results = model.predict(image, classes=[0])
    first_image_results = results[0]

    # Access the bounding boxes data
    boxes = first_image_results.boxes
    # Find the bounding box with the highest confidence
    if len(boxes) > 0:
        # Sort detections by confidence in descending order
        # The .data attribute provides access to the underlying tensor data
        sorted_boxes = sorted(boxes.data, key=lambda x: x[4], reverse=True) # Confidence is at index 4

        # The top bounding box is the first one in the sorted list
        # The format is [x1, y1, x2, y2, confidence, class_index]
        top_box_data = sorted_boxes[0]
        
        # Extract coordinates (xyxy format: top-left x, top-left y, bottom-right x, bottom-right y)
        x1, y1, x2, y2 = map(int, top_box_data[:4])
        
    return (x1, y1, x2, y2)


def main(args):
    print("Loading configuration...")
    config = load_config(path_from_file(__file__, "id_region_config.yaml"))

    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = load_json(args.in_json_path)
    df = join_dataframe(data)
    
    # Expand bbox column into separate x, y, w, h columns
    dataset = CustomImageDataset(df)
    
    print("Loading model...")
    with warnings.catch_warnings():  # Add this line
        warnings.filterwarnings("ignore", category=UserWarning)
        model = load_model(args.model_checkpoint_path, device)
        
    for idx in range(len(dataset)):
        df.iloc[idx]["bbox"] = xyxy_to_xywh(list(get_new_bbox(model, dataset[idx])))
    
    annotations = split_dataframe(df)
    save_json(annotations,args.out_json_path)
    
    # Clean up checkpoint
    if os.path.exists(args.cp_path):
        os.remove(args.cp_path)

    print(
        f"JSON with new bbox: {args.out_json_path}"
    )
    print("All tasks completed successfully!")
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run finetuned YOLOv8n to extract focused images of mew-id region"
    )
    parser.add_argument(
        "in_json_path",
        type=str,
        help="The full path to the filtered IA output json to use as input",
    )
    parser.add_argument(
        "model_checkpoint_path", type=str, help="The full path to the finetuned YOLO model"
    )
    parser.add_argument(
        "out_json_path", type=str, help="The full path to the output json file"
    )
    
    args = parser.parse_args()

    main(args)
    