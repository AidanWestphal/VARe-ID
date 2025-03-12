import argparse
import os
import cv2
import numpy as np
import pandas as pd
import json

import torch
import yaml
from tqdm import tqdm

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from transformers import AutoModel


class MiewIDDataset(Dataset):
    def __init__(self, df, transforms=None, output_label=False):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms

        self.output_label = output_label

        if self.output_label:
            # Aggregate the label columns into a single multi-hot encoded vector
            self.labels = self.df[
                self.label_cols
            ].values  # This creates a NumPy array of shape [num_samples, num_labels]
            self.labels = torch.tensor(
                self.labels, dtype=torch.float32
            )  # Convert to a tensor for PyTorch compatibility

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = get_chip(self.df.loc[index])
        # print(f'Shape of the input image: {img.shape}')    # Print the shape of the image
        if self.transforms:
            img = self.transforms(image=img)["image"]  # Apply transformations
            # print(f'Shape of the transformed image: {img.shape}')
        if self.output_label:
            # Load label data
            target = self.labels[index]
            return img, target
        else:
            return img


def get_img(path):
    return cv2.imread(path)[:, :, ::-1]


def rotate_box(x1, y1, x2, y2, theta):
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]])
    C = np.array([[xm, ym]])
    RA = (A - C) @ R.T + C
    RA = RA.astype(int)
    return RA


def crop_rect(img, rect):
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, np.rad2deg(angle), 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot


def get_chip(row):
    x1 = row["bbox x"]
    y1 = row["bbox y"]
    w = row["bbox w"]
    h = row["bbox h"]
    theta = 0.0
    img = get_img(row["path"]).copy()
    x2 = x1 + w
    y2 = y1 + h
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    return crop_rect(img, ((xm, ym), (x2 - x1, y2 - y1), theta))[0]


def load_config(config_file_path):
    with open(config_file_path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file


def download_model(model_url):
    # Load model directly
    model = AutoModel.from_pretrained(model_url, trust_remote_code=True)
    return model


def get_embeddings(loader, model, device):
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        with tqdm(loader, total=len(loader), desc="Running model...") as pbar:
            for imgs in pbar:
                imgs = imgs.to(device).float()
                img_embeds = model(imgs)
                all_embeddings.append(img_embeds.detach().cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    return all_embeddings


if __name__ == "__main__":
    print("Loading data...")
    parser = argparse.ArgumentParser(
        description="Generate miewid embeddings for database of animal images"
    )
    parser.add_argument(
        "image_dir", type=str, help="The directory where localized images are found"
    )
    parser.add_argument(
        "in_csv_path",
        type=str,
        help="The full path to the ca classifier output csv to use as input",
    )
    parser.add_argument(
        "model_url",
        type=str,
        help="The url to the hugging face model for miewid embeddings",
    )
    parser.add_argument(
        "out_path", type=str, help="The full path to the output json file"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv_path)
    config = load_config("algo/miew_id.yaml")

    print(f"Downloading model {args.model_url}...")
    model = download_model(args.model_url)

    preprocess = Compose(
        [
            Resize(config["img_size"], config["img_size"]),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )

    df["path"] = df["image uuid"].apply(
        lambda x: os.path.join(args.image_dir, x + ".jpg")
    )

    print("Building dataset and loader...")
    ds = MiewIDDataset(df, preprocess)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=config["valid_bs"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device(config["device"])
    embeddings = get_embeddings(dl, model, device)

    print("Building the new annotations...")
    df["miewid"] = embeddings.tolist()
    annots = df.to_dict("records")

    print(f"Saving in json format to {args.out_path}...")
    with open(args.out_path, "w") as f:
        f.write(json.dumps(annots, indent=4))

    print("Done!")
