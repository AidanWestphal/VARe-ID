import argparse
import os
import shutil
import warnings
import json
import pickle
from typing import Any, Callable, Dict

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import yaml
from tqdm import tqdm
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, Sampler

from VAREID.libraries.io.checkpoint import DataLoaderCheckpointManager
from VAREID.libraries.io.format_funcs import load_config, load_json, save_json, split_dataframe, join_dataframe
from VAREID.libraries.utils import path_from_file

# Load configuration
config = load_config(path_from_file(__file__, "viewpoint_classifier_config.yaml"))

class ClassifierDataset(Dataset):
    def __init__(self, df, transforms=None, output_label=False):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms

        self.output_label = output_label
        # self.label_cols = label_cols

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

class ImgClassifier(torch.nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


def get_valid_transforms():
    return Compose(
        [
            Resize(config["img_size"], config["img_size"]),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def reformat_viewpoint(viewpoint):    
    out = ""
    precedence = ["up", "front", "back", "right", "left"]
    for p in precedence:
        if p in viewpoint:
            out += p
    
    return out


def predict_labels_new(dataset, model, device, cp_int, cp_path, batch_size, num_workers):
    model.eval()

    # Store predictions and discrete labels for all samples
    all_preds = []
    all_labels = []

    def get_current_state():
        return {
            "preds": all_preds,
            "labels": all_labels
        }

    manager = DataLoaderCheckpointManager(
        dataset=dataset,
        state_getter=get_current_state,
        checkpoint_interval=cp_int,
        save_path=cp_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    with manager as runner:
        if runner.iteration > 0:
            all_preds = runner.external_state.get("preds", [])
            all_labels = runner.external_state.get("labels", [])

        for imgs in tqdm(runner, desc="Classifying Viewpoint"):
            imgs = imgs.to(device).float()

            with torch.no_grad():
                # Make the prediction
                image_preds = model(imgs)
                preds_sigmoid = torch.sigmoid(
                    image_preds
                )  # Apply sigmoid to get probabilities
                
                batch_preds = preds_sigmoid.detach().cpu().numpy().tolist()
                all_preds.extend(batch_preds)

                # Convert probabilities to labels based on a threshold
                threshold = 0.5
                discrete_labels = (preds_sigmoid > threshold).int()

                B = discrete_labels.shape[0]

                #"back", "front", "left", "right", "up"
                for i in range(B):
                    dl = discrete_labels[i]
                    ps = preds_sigmoid[i]

                    # left / right conflict
                    if dl[2] == 1 and dl[3] == 1:
                        dl[2] = int(ps[2] > ps[3])
                        dl[3] = int(ps[3] > ps[2])

                    # front / back conflict
                    if dl[0] == 1 and dl[1] == 1:
                        dl[0] = int(ps[0] > ps[1])
                        dl[1] = int(ps[1] > ps[0])

                    # no viewpoint at all
                    if dl.sum() == 0:
                        dl[ps.argmax()] = 1

                    # up-only case or upfront case → force left or right
                    if torch.equal(dl, torch.tensor([0, 0, 0, 0, 1], device=dl.device)) or torch.equal(dl, torch.tensor([0, 1, 0, 0, 1], device=dl.device)):
                        if ps[2] > ps[3]:
                            dl[2] = 1
                        elif ps[3] > ps[2]:
                            dl[3] = 1
                        else:    
                            pass
                            
                batch_labels = discrete_labels.detach().cpu().numpy().tolist()
                all_labels.extend(batch_labels)
                

    # Concatenate all batch results
    return np.array(all_preds), np.array(all_labels)


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
    theta = 0.0
    img = cv2.imread(row["image_path"])[:, :, ::-1]
    x1, y1, w, h = row["bbox"]
    x2 = x1 + w
    y2 = y1 + h
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    return crop_rect(img, ((xm, ym), (x2 - x1, y2 - y1), theta))[0]


def main(args):
    original_json = load_json(args.in_json_path)
    annots = join_dataframe(original_json)
    
    if (annots.size == 0):
        raise Exception("Loaded DataFrame is empty, cannot continue pipeline.")


    # Remove rows that are not the desired species
    filtered_annots = annots[
        annots["species"].isin(config["filtered_classes"])
    ]

    # NOTE: MAY REMOVE LATER
    # Split based on bbox_xywh and species criteria
    filtered_test = filtered_annots[
        filtered_annots["bbox"].notna()
    ].reset_index(drop=True)

    other_test = filtered_annots[
        filtered_annots["bbox"].isna()
    ].reset_index(drop=True)

    other_test["viewpoint"] = ""

    # print(f'Filtered dataset is: \n {filtered_test}')
    # print(f'\n Other dataset is: \n {other_test}')

    print("Preparing data for the model...")
    test_ds = ClassifierDataset(filtered_test, transforms=get_valid_transforms())

    print("Setting up the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with warnings.catch_warnings():  # Add this line
        warnings.filterwarnings("ignore", category=UserWarning)
        model = ImgClassifier(
            config["model_arch"], len(config["label_cols"]), pretrained=True
        ).to(device)
        model.load_state_dict(
            torch.load(args.model_checkpoint_path, map_location=device)
        )

    print("Running the model...")
    
    sigmoid_outputs, all_discrete_labels = predict_labels_new(
        dataset=test_ds,
        model=model,
        device=device,
        cp_int=args.cp_freq,
        cp_path=args.cp_path,
        batch_size=config["valid_bs"],
        num_workers=config["num_workers"]
    )
    
    preds_sigmoid_df = pd.DataFrame(
        sigmoid_outputs,
        columns=config["label_cols"]
    )

    filtered_test["viewpoint_sigmoid"] = preds_sigmoid_df.apply(
        lambda row: row.to_dict(), axis=1
    )

    other_test["viewpoint_sigmoid"] = [{} for _ in range(len(other_test))]
    
    print("Processing the model predictions...")
    # Create a DataFrame from the binary labels
    preds_bin = pd.DataFrame(all_discrete_labels, columns=config["label_cols"])

    # Add a new column to the filtered_test DataFrame with the predicted labels
    filtered_test["viewpoint"] = preds_bin.apply(
        lambda row: ", ".join(row.index[row == 1]), axis=1
    )

    # Concatenate filtered_test and other_test dataframes
    final_output = pd.concat([filtered_test, other_test])

    # Reformat viewpoints to singular words
    final_output["viewpoint"] = final_output["viewpoint"].apply(
        lambda x: reformat_viewpoint(x)
    )

    # Save the updated DataFrame to a new JSON file
    viewpoint_dir = os.path.dirname(args.out_json_path)

    # Logic updated: Do not delete directory if resuming
    if not os.path.exists(viewpoint_dir):
        os.makedirs(viewpoint_dir, exist_ok=True)

    print("Saving the results...")
    final_json = split_dataframe(final_output)
    save_json(final_json, args.out_json_path)

    # Clean up checkpoint
    if os.path.exists(args.cp_path):
        os.remove(args.cp_path)

    print("Done!")


if __name__ == "__main__":
    print("Loading data...")
    parser = argparse.ArgumentParser(
        description="Run viewpoint classifier for database of animal images"
    )
    parser.add_argument(
        "in_json_path",
        type=str,
        help="The annotations json file to add viewpoints to",
    )
    parser.add_argument(
        "model_checkpoint_path", type=str, help="The full path to the model checkpoint"
    )
    parser.add_argument(
        "cp_freq", type=int, help="The checkpoint frequency for safe exiting"
    )
    parser.add_argument(
        "cp_path", type=str, help="The checkpoint path for safe exiting"
    )
    parser.add_argument(
        "out_json_path", type=str, help="The full path to the output json file"
    )
    args = parser.parse_args()
    main(args)