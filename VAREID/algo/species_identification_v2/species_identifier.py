import argparse
import os
import shutil
import tempfile
import warnings
from pathlib import Path
import json

import pandas as pd
import yaml
from tqdm import tqdm
from PIL import Image

from bioclip.predict import TreeOfLifeClassifier, Rank

from VAREID.libraries.io.checkpoint import CheckpointManager
from VAREID.libraries.utils import path_from_file
from VAREID.libraries.io.format_funcs import (
    load_config,
    load_json,
    save_json,
    split_dataframe,
    join_dataframe,
)

warnings.filterwarnings("ignore")


def run_pyBioclip(classifier, df, species_list, top_k, cp_int, cp_path):

    predicted_labels = []
    predicted_scores = []
    prediction_dicts = []

    # STATE GETTER FOR CHECKPOINTING
    state_getter = lambda: {
        "labels": predicted_labels,
        "scores": predicted_scores,
        "pred_dicts": prediction_dicts,
    }

    with CheckpointManager(df.iterrows(), state_getter, cp_int, cp_path, len(df)) as cpdata:

        if cpdata.iteration > 0:
            predicted_labels = cpdata.external_state["labels"]
            predicted_scores = cpdata.external_state["scores"]
            prediction_dicts = cpdata.external_state["pred_dicts"]

        for _, row in tqdm(cpdata, initial=cpdata.iteration, desc="Identifying Species"):
            x0, y0, w, h = row["bbox"]

            original_image = Image.open(row["image_path"])
            cropped_image = original_image.crop((x0, y0, x0 + w, y0 + h))

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.close()
            cropped_image.save(temp_file.name)

            predictions = classifier.predict(
                temp_file.name,
                rank=Rank.SPECIES,
                k=top_k,
            )

            os.remove(temp_file.name)

            pred_dict = {p["species"]: float(p["score"]) for p in predictions}
            prediction_dicts.append(pred_dict)

            allowed_preds = {
                s: pred_dict[s] for s in species_list if s in pred_dict
            }

            if allowed_preds:
                best_species = max(allowed_preds, key=allowed_preds.get)
                best_score = allowed_preds[best_species]
            else:
                best_pred = max(predictions, key=lambda x: x["score"])
                best_species = best_pred["species"]
                best_score = best_pred["score"]

            predicted_labels.append(best_species)
            predicted_scores.append(best_score)

    category_ids, _ = pd.factorize(predicted_labels)

    df["species"] = predicted_labels
    df["species_score"] = predicted_scores
    df["category_id"] = category_ids
    df["prediction_dict"] = [json.dumps(d) for d in prediction_dicts]

    return df


def pyBioCLIP(df, species_list, top_k, cp_int, cp_path):
    classifier = TreeOfLifeClassifier()
    return run_pyBioclip(classifier, df, species_list, top_k, cp_int, cp_path)


def main(args):
    # Load config
    config = load_config(path_from_file(__file__, "species_identifier_config.yaml"))

    if os.path.exists(args.si_dir):
        print("Removing Previous Instance of Experiment")
        shutil.rmtree(args.si_dir)

    print("Creating Experiment Directory ...")
    os.makedirs(args.si_dir, exist_ok=True)

    print("Running pyBioCLIP ...")

    data = load_json(args.in_json_path)
    df = join_dataframe(data)

    # Build allowed species list from config
    species_map = config["species_map"]
    species_list = list(species_map.values())

    top_k = config["prediction"]["top_k"]

    df = pyBioCLIP(df, species_list, top_k, args.cp_freq, args.cp_path)

    print("pyBioCLIP Completed ...")

    prediction_dir = os.path.dirname(args.out_json_path)
    shutil.rmtree(prediction_dir, ignore_errors=True)
    os.makedirs(prediction_dir, exist_ok=True)

    if df.size == 0:
        raise Exception("Species identifier found nothing, cannot continue pipeline.")

    print("Saving ALL Predictions as JSON ...")
    annotations = split_dataframe(df)
    save_json(annotations, args.out_json_path)

    if os.path.exists(args.cp_path):
        os.remove(args.cp_path)

    print("Completed Successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Classify species for annotations"
    )
    parser.add_argument(
        "in_json_path",
        type=str,
        help="The full path to the annotations json file",
    )
    parser.add_argument(
        "si_dir",
        type=str,
        help="The directory to install bioCLIP within",
    )
    parser.add_argument(
        "out_json_path",
        type=str,
        help="The full path to the output json file",
    )
    parser.add_argument(
        "cp_freq",
        type=int,
        help="The checkpoint frequency for safe exiting",
    )
    parser.add_argument(
        "cp_path",
        type=str,
        help="The checkpoint path for safe exiting",
    )
    args = parser.parse_args()
    main(args)
