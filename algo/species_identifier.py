import os
import ast
import sys
import yaml
import torch
import shutil
import argparse
import tempfile
import warnings
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def load_config(config_file_path):
    with open(config_file_path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file


def load_annotations_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df


def clone_pyBioCLIP_from_github(directory, repo_url):
    if not os.path.exists(directory) or not os.listdir(directory):
        print(f"Cloning repository {repo_url} into {directory}...")
        subprocess.run(["git", "clone", repo_url, directory], check=True)
        print("Repository cloned successfully...")
    else:
        print("Repository already cloned...")


def install_pyBioCLIP_from_directory(directory):
    try:
        print(f"Installing package from {directory}...")
        subprocess.run([sys.executable, "-m", "pip", "install", directory], check=True)
        print("Package installed successfully...")
    except subprocess.CalledProcessError as e:
        print(f"Error installing package from {directory}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_bioCLIP(url, target_dir):
    try:
        result = subprocess.run(
            ["pip", "install", f"git+{url}", "--target", target_dir],
            capture_output=True,
            check=True,
        )
        print(f"Successfully installed from {url} to {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing from {url} to {target_dir}:")
        print(e.stderr.decode("utf-8"))


def run_pyBioclip(bioclip_classifier, image_dir, df):

    predicted_labels = []
    predicted_scores = []

    for index, row in tqdm(df.iterrows()):

        x0 = row["bbox x"]
        y0 = row["bbox y"]
        w = row["bbox w"]
        h = row["bbox h"]
        image_filename = row["image uuid"]

        image_filepath = os.path.join(image_dir, f"{image_filename}")
        original_image = Image.open(image_filepath)
        cropped_image = original_image.crop((x0, y0, x0 + w, y0 + h))

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.close()
        cropped_image.save(temp_file.name)

        predictions = bioclip_classifier.predict(temp_file.name)

        top_prediction = max(predictions, key=lambda x: x["score"])
        predicted_label = top_prediction["classification"]
        pred_conf_score = top_prediction["score"]

        predicted_labels.append(predicted_label)
        predicted_scores.append(pred_conf_score)
        os.remove(temp_file.name)

    df["species_prediction"] = predicted_labels
    df["species_pred_score"] = predicted_scores

    return df


def pyBioCLIP(labels, image_dir, df):

    classifier = CustomLabelsClassifier(labels)
    df = run_pyBioclip(classifier, image_dir, df)

    return df


def simplify_species(species_name, category_map):
    for key, value in category_map.items():
        if key in species_name:
            return value
    return None


def postprerocess_dataframe(df):

    # this is only when dectection has filter (ground truth)
    # category_map_true = {"zebra_grevys": 0, "zebra_plains": 1, "neither": 2}
    # df["species_true_simple"] = df["annot species"].apply(
    #     lambda x: simplify_species(x, category_map_true)
    # )

    category_map_pred = {"grevy's zebra": 0, "plains zebra": 1, "neither": 2}
    df["species_pred_simple"] = df["species_prediction"].apply(
        lambda x: simplify_species(x, category_map_pred)
    )

    return df


if __name__ == "__main__":

    # Loading Configuration File ...
    config = load_config("algo/species_identifier_drive.yaml")

    # Setting up Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Detect bounding boxes for database of animal images"
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
        "si_dir", type=str, help="The directory to install bioCLIP within"
    )
    parser.add_argument(
        "out_csv_path", type=str, help="The full path to the output csv file"
    )
    args = parser.parse_args()

    images = Path(args.image_dir)

    if os.path.exists(args.si_dir):
        print("Removing Previous Instance of Experiment")
        shutil.rmtree(args.si_dir)

    print("Creating Experiment Directory ...")
    os.makedirs(args.si_dir, exist_ok=True)

    bioCLIP_dir = os.path.join(args.si_dir, config["bioclip_dirname"])
    pyBioCLIP_url = config["github_bioclip_url"]

    print("Cloning & Installing pyBioCLIP ...")
    clone_pyBioCLIP_from_github(bioCLIP_dir, pyBioCLIP_url)
    install_pyBioCLIP_from_directory(bioCLIP_dir)
    from bioclip import CustomLabelsClassifier

    print("pyBioCLIP Installation Completed ....")

    print("Running pyBioCLIP ...")
    labels = config["custom_labels"]
    df = load_annotations_from_csv(args.in_csv_path)
    df = pyBioCLIP(labels, images, df)
    print("pyBioCLIP Completed ...")

    print("Post-Processing ...")
    df = postprerocess_dataframe(df)
    print("Post-Processing Completed ...")

    prediction_dir = os.path.dirname(args.out_csv_path)
    shutil.rmtree(prediction_dir, ignore_errors=True)
    os.makedirs(prediction_dir, exist_ok=True)

    print("Saving ALL Predictions as CSV ...")

    df.to_csv(args.out_csv_path, index=False)

    print("Completed Successfully!")
