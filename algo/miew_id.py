import argparse
import pickle
import cv2
import numpy as np
import pandas as pd
import json
import os

import torch
import yaml
from tqdm import tqdm

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from transformers import AutoModel


def load_json(file_path):
    """Load JSON data from the given file."""
    with open(file_path, "r") as file:
        return json.load(file)


class MiewIDDataset(Dataset):
    def __init__(self, df, images, transforms=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.images = images
        self.transforms = transforms

        # Build a custom mapping for UUIDS s.t. we have unique integer labels
        uuid_set = set(self.df["uuid"].values.tolist())

        self.uuid_to_id = {uuid: i for i, uuid in enumerate(uuid_set)}
        self.id_to_uuid = {i: uuid for i, uuid in enumerate(uuid_set)}

        self.labels = [*map(self.uuid_to_id.get, self.df["uuid"].values.tolist())]
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = get_chip(self.df.loc[index], self.images)
        # print(f'Shape of the input image: {img.shape}')    # Print the shape of the image
        if self.transforms:
            img = self.transforms(image=img)["image"]  # Apply transformations
            label = self.id_to_uuid[self.labels[index].item()]
            return img, label


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


def get_chip(row, images):
    x1 = row["bbox"][0]
    y1 = row["bbox"][1]
    w = row["bbox"][2]
    h = row["bbox"][3]
    theta = 0.0
    img_uuid = row["image_uuid"]
    image_path = [x["file_name"] for x in images if x["uuid"] == img_uuid][0]
    img = cv2.imread(os.path.join(args.image_dir, image_path))[:, :, ::-1]
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
    all_uuids = []

    with torch.no_grad():
        with tqdm(loader, total=len(loader), desc="Running model...") as pbar:
            for imgs, uuids in pbar:
                imgs = imgs.to(device).float()
                img_embeds = model(imgs)
                all_embeddings.append(img_embeds.detach().cpu())
                all_uuids.append(uuids)
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_uuids = [i for sub in all_uuids for i in sub]

    return (all_embeddings, all_uuids)


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
        "out_file", type=str, help="The full path to the output pickle file"
    )
    args = parser.parse_args()

    data = load_json(args.in_csv_path)
    df = pd.DataFrame(data["annotations"])
    images = data["images"]
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

    print("Building dataset and loader...")
    ds = MiewIDDataset(df, images, preprocess)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=config["valid_bs"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device(config["device"])
    embeddings = get_embeddings(dl, model, device)

    print(f"Saving embeddings file {args.out_file}...")
    with open(args.out_file, "wb") as f:
        pickle.dump(embeddings, f)

    print("Done!")
