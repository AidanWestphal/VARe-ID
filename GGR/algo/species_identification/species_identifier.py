import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
import json
import ast

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from bioclip import CustomLabelsClassifier

warnings.filterwarnings("ignore")
from PIL import Image

from algo.util.io.format_funcs import load_config, load_json, save_json, split_dataframe, join_dataframe


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


def run_pyBioclip(bioclip_classifier, df):

    predicted_labels = []
    predicted_scores = []

    for _, row in tqdm(df.iterrows()):
        x0, y0, w, h = row["bbox"]

        original_image = Image.open(row["image_path"])
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

    category_ids, _ = pd.factorize(predicted_labels)

    df["species"] = predicted_labels
    df["species_score"] = predicted_scores
    df["category_id"] = category_ids

    return df


def pyBioCLIP(labels, df):

    classifier = CustomLabelsClassifier(labels)
    df = run_pyBioclip(classifier, df)

    return df


def simplify_species(species_name, category_map):
    for key, value in category_map.items():
        if key in species_name:
            return value
    return None

def main(args):
    """
    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/species_identifier.py

    Example:
        >>> from argparse import Namespace
        >>> args = Namespace(
        ...     image_dir="test_imgs",
        ...     in_csv_path="temp/annots/filtered_annots.csv",
        ...     si_dir="temp/bioclip",
        ...     out_csv_path="temp/species_classifier/species_classifier_output.csv"
        ... )
        >>> main(args) # doctest: +ELLIPSIS
            Removing Previous Instance of Experiment
            Creating Experiment Directory ...
            Cloning & Installing pyBioCLIP ...
            Cloning repository https://... into temp/bioclip/BioCLIP/...
            Repository cloned successfully...
            Installing package from temp/bioclip/BioCLIP/...
            Package installed successfully...
            pyBioCLIP Installation Completed ....
            Running pyBioCLIP ...
            pyBioCLIP Completed ...
            Post-Processing ...
            Post-Processing Completed ...
            Saving ALL Predictions as CSV ...
            Completed Successfully!
    """

    # Loading Configuration File ...
    config = load_config("algo/species_identifier.yaml")

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

    print("pyBioCLIP Installation Completed ....")

    print("Running pyBioCLIP ...")
    labels = config["custom_labels"]
    data = load_json(args.in_csv_path)
    df = join_dataframe(data)
    df = pyBioCLIP(labels, df)
    print("pyBioCLIP Completed ...")

    prediction_dir = os.path.dirname(args.out_csv_path)
    shutil.rmtree(prediction_dir, ignore_errors=True)
    os.makedirs(prediction_dir, exist_ok=True)

    print("Saving ALL Predictions as JSON ...")
    annotations = split_dataframe(df)
    save_json(annotations,args.out_csv_path)

    print("Completed Successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Detect bounding boxes for database of animal images"
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
    main(args)
