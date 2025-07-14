import argparse
import os
import shutil
import warnings
import json

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import yaml
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from GGR.util.io.format_funcs import load_config, load_json, save_json, split_dataframe, join_dataframe
from GGR.util.utils import path_from_file

# Load configuration
config = load_config(path_from_file(__file__, "viewpoint_classifier.yaml"))

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


def predict_labels_new(test_loader, model, device):
    model.eval()

    # Store predictions and discrete labels for all samples
    all_preds = []
    all_discrete_labels = []

    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device).float()

            # Make the prediction
            image_preds = model(imgs)
            preds_sigmoid = torch.sigmoid(
                image_preds
            )  # Apply sigmoid to get probabilities
            all_preds.append(preds_sigmoid.detach().cpu())

            # Convert probabilities to labels based on a threshold
            threshold = 0.5
            discrete_labels = (preds_sigmoid > threshold).int()
            all_discrete_labels.append(discrete_labels.detach().cpu())

    # Concatenate all batch results
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_discrete_labels = torch.cat(all_discrete_labels, dim=0).numpy()

    return all_preds, all_discrete_labels


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
    """
    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/viewpoint_classifier.py

    Example:
        >>> from argparse import Namespace
        >>> args = Namespace(
        ...     image_dir="test_imgs",
        ...     in_csv_path="temp/species_classifier/species_classifier_output.csv",
        ...     model_checkpoint_path="test_dataset/viewpoint_trained_model.pth",
        ...     out_csv_path="temp/viewpoint/viewpoint_output.csv"
        ... )
        >>> main(args) # doctest: +ELLIPSIS
        Index(['annot uuid', 'image uuid', 'image fname', 'bbox x', 'bbox y', 'bbox w',
            'bbox h', 'bbox pred score', 'category id', 'species_prediction',
            'species_pred_score', 'species_pred_simple'],
            dtype='object')
        Preparing data for the model...
        Setting up the model...
        Running the model...
        Processing the model predictions...
        Removing Previous Instance of Experiment...
        Saving the results...
        Done!
    """

    original_json = load_json(args.in_csv_path)
    annots = join_dataframe(original_json)

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
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config["valid_bs"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=False,
    )

    print("Setting up the model...")
    device = torch.device(config["device"])
    with warnings.catch_warnings():  # Add this line
        warnings.filterwarnings("ignore", category=UserWarning)
        model = ImgClassifier(
            config["model_arch"], len(config["label_cols"]), pretrained=True
        ).to(device)
        model.load_state_dict(
            torch.load(args.model_checkpoint_path, map_location=config["device"])
        )

    print("Running the model...")
    _, all_discrete_labels = predict_labels_new(test_loader, model, device)

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

    # Save the updated DataFrame to a new CSV file
    viewpoint_dir = os.path.dirname(args.out_csv_path)

    if os.path.exists(viewpoint_dir):
        print("Removing Previous Instance of Experiment...")
        shutil.rmtree(viewpoint_dir)

    print("Saving the results...")
    os.makedirs(viewpoint_dir, exist_ok=True)

    final_json = split_dataframe(final_output)
    save_json(final_json, args.out_csv_path)

    print("Done!")


if __name__ == "__main__":
    print("Loading data...")
    parser = argparse.ArgumentParser(
        description="Run viewpoint classifier for database of animal images"
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
