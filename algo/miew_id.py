import argparse
import os
import pickle
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
    def __init__(self, df, transforms=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms

        # Build a custom mapping for UUIDS s.t. we have unique integer labels
        uuid_set = set(self.df["uuid"].values.tolist())

        self.uuid_to_id = { uuid: i for i, uuid in enumerate(uuid_set) }
        self.id_to_uuid = { i: uuid for i, uuid in enumerate(uuid_set) }

        self.labels = [*map(self.uuid_to_id.get, self.df["uuid"].values.tolist())]
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = get_chip(self.df.loc[index])
        # print(f'Shape of the input image: {img.shape}')    # Print the shape of the image
        if self.transforms:
            img = self.transforms(image=img)["image"]  # Apply transformations
            label = self.id_to_uuid[self.labels[index].item()]
            return img, label

        

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def rotate_box(x1, y1, x2, y2, theta):
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    h = int(y2 - y1)
    w = int(x2 - x1)
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
    all_uuids = []

    with torch.no_grad():
        with tqdm(loader,total=len(loader),desc="Running model...") as pbar:
            for imgs, uuids in pbar:
                imgs = imgs.to(device).float()
                img_embeds = model(imgs)
                all_embeddings.append(img_embeds.detach().cpu())
                all_uuids.append(uuids)
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_uuids = [i for sub in all_uuids for i in sub]

    return (all_embeddings, all_uuids)

def format_for_lca(annots):
    # Aggregate mapping category id to species name
    categories = {}
    # Aggregate mapping image UUID to fname
    images = {}
    # List of properly formatted annotations
    formatted_annots = []
    for a in tqdm(annots,desc="Reformatting annotations..."):
        categories[a["species_pred_simple"]] = a["species_prediction"]
        images[a["image uuid"]] = a["image fname"]
        
        formatted_annots.append(
            {
            "uuid": a["uuid"],
            "image_uuid": a["image uuid"],
            "bbox": [a["bbox x"], a["bbox y"], a["bbox w"], a["bbox h"]],
            "viewport": a["predicted_viewpoint"],
            "tracking_id": 0, # TODO: PLACEHOLDER
            "confidence": a["bbox pred score"],
            "detection_class": a["category id"],
            "species": a["species_prediction"],
            "CA_score": a["CA_score"],
            "category_id": a["species_pred_simple"],
            }
        )
    # Reformat into a combined json data dictionary
    json_data = {
        "categories": [ {"id": id, "species": spec} for id, spec in categories.items() ],
        "images": [ {"file_name": fname, "uuid": uuid} for uuid, fname in images.items() ],
        "annotations": formatted_annots
    }
    return json_data


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
        help="The url to the hugging face model for miewid embeddings"
    )
    parser.add_argument(
        "out_json", type=str, help="The full path to the output json file"
    )
    parser.add_argument(
        "out_pickle", type=str, help="The full path to the output pickle file"
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
        ])

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
    annots = df.to_dict('records')
    json_data = format_for_lca(annots)

    print(f"Saving output files {args.out_json} and {args.out_pickle}...")
    with open(args.out_json, "w") as f:
        f.write(json.dumps(json_data,indent=4))
    with open(args.out_pickle, "wb") as f:
        pickle.dump(embeddings,f)

    print("Done!")
    