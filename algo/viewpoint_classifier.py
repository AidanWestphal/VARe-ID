import argparse
import ast
import os
import shutil
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import yaml
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset
from tqdm import tqdm

# Load configuration
with open("algo/viewpoint_classifier.yaml", "r") as file:
    config = yaml.safe_load(file)


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

    def load_image(self, img_path):
        # Load image from the file system; placeholder function
        # You should replace this with actual image loading logic
        img = np.random.rand(
            224, 224, 3
        )  # Placeholder: Replace with actual image loading
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

#
# def get_chip(row):
#     # box = ast.literal_eval(row['bbox'])
#     box = row["bbox"]
#     theta = 0.0
#     img = get_img(row["path"]).copy()
#     x1, y1, w, h = box
#     x2 = x1 + w
#     y2 = y1 + h
#     xm = (x1 + x2) // 2
#     ym = (y1 + y2) // 2
#     return crop_rect(img, ((xm, ym), (x2 - x1, y2 - y1), theta))[0]

def get_chip(row):
    box = row["bbox_xywh"]  # Changed from bbox to bbox_xywh
    theta = 0.0
    img = get_img(row["path"]).copy()
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    return crop_rect(img, ((xm, ym), (x2 - x1, y2 - y1), theta))[0]


if __name__ == "__main__":
    print("Loading data...")
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

    original_csv = pd.read_csv(args.in_csv_path)

    print(original_csv.columns)
    # Remove rows that are not the correct species
    # is this needed? this would require ground truth
    filtered_csv = original_csv[
        # original_csv["species_true_simple"].isin(config["filtered_classes"])
        original_csv["species_pred_simple"].isin(config["filtered_classes"])
    ]
    # Append image_dir to the 'image fname' column
    # filtered_csv["path"] = filtered_csv["image uuid"+"jpg"].apply(
    #     lambda x: os.path.join(args.image_dir, x)
    # )

    filtered_csv["path"] = filtered_csv["image uuid"].apply(
        lambda x: os.path.join(args.image_dir, x + ".jpg")
    )
    
    # # Create a single 'bbox' column from the four bbox columns
    # filtered_csv["bbox"] = list(
    #     zip(
    #         filtered_csv["bbox x"],
    #         filtered_csv["bbox y"],
    #         filtered_csv["bbox w"],
    #         filtered_csv["bbox h"],
    #     )
    # )

    import ast

    # First convert string bbox to list
    filtered_csv['bbox_xyxy'] = filtered_csv['bbox'].apply(ast.literal_eval)


    # Convert xyxy to xywh
    def xyxy_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1
        y = y1
        return [x, y, w, h]


    filtered_csv['bbox_xywh'] = filtered_csv['bbox_xyxy'].apply(xyxy_to_xywh)

    #
    # # Split the original dataframe into two based on the filtering criteria
    # filtered_test = filtered_csv[
    #     (filtered_csv[["bbox x", "bbox y", "bbox w", "bbox h"]].notna().all(axis=1))
    #     & (filtered_csv["annot species"] == config["species"])
    # ].reset_index(drop=True)
    # other_test = filtered_csv[
    #     ~(filtered_csv[["bbox x", "bbox y", "bbox w", "bbox h"]].notna().all(axis=1))
    #     | (filtered_csv["annot species"] != config["species"])
    # ].reset_index(drop=True)
    # # Add a 'predicted_viewpoint' column filled with NaNs to the other_test dataframe
    # other_test["predicted_viewpoint"] = np.nan

    # Split based on bbox_xywh and species criteria
    filtered_test = filtered_csv[
        filtered_csv['bbox_xywh'].notna() &
        (filtered_csv["species_prediction"] == config["species"])
        ].reset_index(drop=True)

    other_test = filtered_csv[
        filtered_csv['bbox_xywh'].isna() |
        (filtered_csv["species_prediction"] != config["species"])
        ].reset_index(drop=True)

    other_test["predicted_viewpoint"] = np.nan


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
        scaler = GradScaler()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["T_0"],
            T_mult=1,
            eta_min=float(config["min_lr"]),
            last_epoch=-1,
        )
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

    print("Running the model...")
    _, all_discrete_labels = predict_labels_new(test_loader, model, device)

    print("Processing the model predictions...")
    # Create a DataFrame from the binary labels
    preds_bin = pd.DataFrame(all_discrete_labels, columns=config["label_cols"])
    # print(f'The predictions are: {preds_bin}\n')

    # Add a new column to the filtered_test DataFrame with the predicted labels
    filtered_test["predicted_viewpoint"] = preds_bin.apply(
        lambda row: ", ".join(row.index[row == 1]), axis=1
    )

    # Concatenate filtered_test and other_test dataframes
    final_output = pd.concat([filtered_test, other_test])

    # Save the updated DataFrame to a new CSV file
    viewpoint_dir = os.path.dirname(args.out_csv_path)

    if os.path.exists(viewpoint_dir):
        print("Removing Previous Instance of Experiment...")
        shutil.rmtree(viewpoint_dir)

    print("Saving the results...")
    os.makedirs(viewpoint_dir, exist_ok=True)
    final_output.to_csv(args.out_csv_path, index=False)

    print("Done!")
