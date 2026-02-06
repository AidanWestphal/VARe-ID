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

from tqdm import tqdm
import cv2

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

def apply_nms(df, iou_threshold):
    df = df.sort_values("CA_score", ascending=False)
    
    boxes = np.array([xywh_to_xyxy(bbox) for bbox in df["bbox"]])
    # scores = df["CA_score"].values
    
    #filter by aspect ratio
    heights  = boxes[:, 3] - boxes[:, 1]
    widths = boxes[:, 2] - boxes[:, 0]
    scores = widths/heights
    
    boxes = torch.as_tensor(boxes).float()
    scores = torch.as_tensor(scores).float()
    
    keep_positions = nms(boxes, scores, iou_threshold)
    
    keep_positions = keep_positions.cpu().numpy()
    kept_index_labels = df.index[keep_positions]
    
    removed_index_labels = df.index.difference(kept_index_labels)
    
    if len(removed_index_labels) > 0:
        df.loc[removed_index_labels, 'annotations_census'] = False
        
    return df

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

        # # Flip the image if left viewpoint
        # if "left" in self.img_data.iloc[idx]["viewpoint"]:
        #     image = F.hflip(image)

        # if self.transform:
        #     image = self.transform(image)

        return image
    
    def get_path(self, idx):
        return self.img_data.iloc[idx]["image_path"]
    
    def get_original_img(self, idx):
        # Read image as PIL Image
        image = Image.open(self.img_data.iloc[idx]["image_path"]).convert("RGB")

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

def get_new_bbox(model, image, conf=0.0, x_scale=1, y_scale=1):
    results = model.predict(image, classes=[0], conf=conf, verbose=False)
    first_image_results = results[0]

    # Access the bounding boxes data
    boxes = first_image_results.boxes
    # Find the bounding box with the highest confidence
    
    x1, y1, x2, y2 = -1, -1, -1, -1
    
    if len(boxes) > 0:
        # Sort detections by confidence in descending order
        # The .data attribute provides access to the underlying tensor data
        sorted_boxes = sorted(boxes.data, key=lambda x: x[4], reverse=True) # Confidence is at index 4

        # The top bounding box is the first one in the sorted list
        # The format is [x1, y1, x2, y2, confidence, class_index]
        top_box_data = sorted_boxes[0]
        
        # Extract coordinates (xyxy format: top-left x, top-left y, bottom-right x, bottom-right y)
        x1, y1, x2, y2 = map(int, top_box_data[:4])
        
    if x_scale != 1:
        w = x2-x1
        x_m = (x1+x2)//2
        x1 = x_m - int((w/2)*x_scale)
        x2 = x_m + int((w/2)*x_scale)
    if y_scale != 1:
        h = y2-y1
        y_m = (y1+x2)//2
        y1 = y_m - int((h/2)*y_scale)
        y2 = y_m + int((h/2)*y_scale)
        
    return (x1, y1, x2, y2)

def calculate_zebra_clarity(pil_crop, target_width=300):
    """
    Calculates a normalized clarity score specifically for zebra patterns.
    """
    # Convert PIL to OpenCV (BGR)
    open_cv_crop = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_crop, cv2.COLOR_BGR2GRAY)
    
    # Resize to standard width to normalize stripe frequency
    h, w = gray.shape
    aspect_ratio = h / w
    gray = cv2.resize(gray, (target_width, int(target_width * aspect_ratio)))
    
    # Calculate Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()

    edges = cv2.Canny(gray, 100, 200)
    
    # Normalization: Sharpness perception scales with brightness
    # Dividing by mean intensity helps separate dark/shadow blur from bright sharp shots
    mean_intensity = np.mean(gray) + 1e-6
    normalized_score = variance / mean_intensity
    
    edge_count = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = (edge_count / total_pixels) * 100
        
    return (normalized_score + edge_density)/2
    
    # im_dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    # im_dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    # return np.mean(np.square(im_dx) + np.square(im_dy))

def point_contained(point, bbox):
    x, y = point
    return bbox[0] < x < bbox[0]+bbox[2] and bbox[1] < y < bbox[1]+bbox[3]

def main(args):
    print("Loading configuration...")
    config = load_config(path_from_file(__file__, "id_region_config.yaml"))

    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = load_json(args.in_json_path)
    df = join_dataframe(data)
    df['annotations_census'] = False
    
    # Expand bbox column into separate x, y, w, h columns
    dataset = CustomImageDataset(df)
    
    print("Loading model...")
    with warnings.catch_warnings():  # Add this line
        warnings.filterwarnings("ignore", category=UserWarning)
        model = load_model(args.model_checkpoint_path, device)
        
    for idx in tqdm(range(len(dataset))):
        img = dataset[idx]
        new_bbox = get_new_bbox(model, img, conf=config['confidence_threshold'], x_scale=config['x_scale'], y_scale=config['y_scale'])
        new_bbox = xyxy_to_xywh(list(new_bbox))
        center_x, center_y = img.size
        center_x /= 2
        center_y /= 2
        center = (center_x, center_y)
        
        if new_bbox[3] > 0:
            aspect_ratio = new_bbox[2]/new_bbox[3]
        else:
            aspect_ratio = 0
        
        if new_bbox[0] != -1:
            id_region_crop = dataset[idx].crop(xywh_to_xyxy(new_bbox))
            clarity_score = calculate_zebra_clarity(id_region_crop)
            df.at[idx, "clarity_score"] = clarity_score
            
            original_x1, original_y1 = df.iloc[idx]["bbox"][0], df.iloc[idx]["bbox"][1]
            adjusted_bbox = [new_bbox[0]+original_x1, new_bbox[1]+original_y1, new_bbox[2], new_bbox[3]]
            df.at[idx, "bbox"] = adjusted_bbox
            if aspect_ratio > config['AR_threshold'] and clarity_score >= config.get('clarity_threshold', 0):
                df.at[idx, 'annotations_census'] = True
    
    # Step 3: Apply NMS
    df = df.groupby("image_path").apply(lambda x: apply_nms(x, config["NMS_threshold"]))

    annotations = split_dataframe(df)
    save_json(annotations,args.out_json_path)
    
    # # Clean up checkpoint
    # if os.path.exists(args.cp_path):
    #     os.remove(args.cp_path)

    print(
        f"JSON with new bbox: {args.out_json_path}"
    )
    print("All tasks completed successfully!")
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run finetuned YOLOv8n to extract focused images of id region"
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
    