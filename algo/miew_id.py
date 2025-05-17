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

import torch.nn.functional as F

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
    img = cv2.imread(row["image path"])[:, :, ::-1]
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


def topk_precision(names, distmat, k, names_db=None, return_matches=False):
    """Computes precision at k given a distance matrix.
    Assumes the distance matrix is square and does one-vs-all evaluation"""
    # assert distmat.shape[0] == distmat.shape[1], "Distance matrix must be square"

    if names_db is None:
        names_db = names

    output = torch.Tensor(distmat[:, :]) * -1
    y = torch.Tensor(names[:]).squeeze(0)
    ids_tensor = torch.Tensor(names_db)

    ranks = list(range(1, k + 1))
    max_k = k

    topk_idx = output.topk(max_k)[1][:, :]  ###
    topk_names = ids_tensor[topk_idx]

    match_mat = topk_names == y[:, None].expand(topk_names.shape)
    scores = []
    for k in ranks:
        match_mat_k = match_mat[:, :k]
        score = match_mat_k.float().mean(axis=1)
        scores.append(score.tolist())

    if return_matches:
        return scores, match_mat, topk_idx, topk_names
    else:
        return scores


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def avg_distance(distmat, labels):
    """Calculates the average distance between a selection of points (by labels).

    Args:
        distmat: 2-D distance matrix.
        labels: List of column/row numbers to consider.

    Returns:
        float: average distance
    """
    # Splice matrix by column/row values
    splice = distmat[labels, :]
    splice = splice[:, labels]
    # Edge case: only one entity, distance is 0
    if splice.shape[0] <= 1:
        return 0
    # Calculate average value of upper triangular section, ignoring main diagonal (don't count edges to self)
    important = torch.triu(splice, diagonal=1)
    indices = torch.triu_indices(splice.shape[0], splice.shape[1], 1)
    num_pts = indices.shape[1]
    sum = torch.sum(important)

    return sum / num_pts


def eval_metrics(embeddings, df, config, dir):
    # STEP 1: Get distmat and labels (map annot uuid to individual id)
    distmat = cosine_distance(torch.Tensor(embeddings[0]), torch.Tensor(embeddings[0]))
    names_uuids = pd.DataFrame({"uuid": embeddings[1]})
    names_df = pd.merge(df, names_uuids, on=["uuid"])
    names_list = [
        "IID: " + str(d["individual_id"]) + " Viewpoint: " + d["viewpoint"]
        for _, d in names_df.iterrows()
    ]
    names = torch.zeros(len(names_list)) - 1
    # ASSIGN INTEGERS FOR ABOVE NAME LIST
    for i in range(len(names_list)):
        # If this is a new label, assign it the current index
        if names[i] == -1:
            names[i] = i
            # Assign all following matches to this index
            for j in range(i + 1, len(names_list)):
                if names_list[i] == names_list[j]:
                    names[j] = i

    # STEP 2: Read rest of params
    k = config["k"]
    log_file = os.path.join(dir, config["eval_file"])

    # STEP 3: Evaluate functions for all pairs
    scores = torch.Tensor(topk_precision(names, distmat, k))
    scores = scores[-1]
    avg_prec = {}
    avg_dist = {}
    for i in names:
        i = int(i.item())
        # If this hasn't been assigned yet...
        if i not in avg_prec.keys():
            indices = torch.where(names == i)[0]
            avg_dist[i] = avg_distance(distmat, indices)
            avg_prec[i] = torch.mean(scores[indices])

    # STEP 4: SAVE
    with open(log_file, "w") as f:
        for i in avg_prec.keys():
            f.write(
                f"{names_list[i]}: Average Precision: {avg_prec[i]} Average Distance: {avg_dist[i]}\n"
            )


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

    if "eval_file" in config.keys():
        print("Evaluating Top-K miewid precision...")
        sep = args.out_file.rfind("/")
        dir = args.out_file[:sep]
        eval_metrics(embeddings, df, config, dir)

    print(f"Saving embeddings file {args.out_file}...")
    with open(args.out_file, "wb") as f:
        pickle.dump(embeddings, f)

    print("Done!")
