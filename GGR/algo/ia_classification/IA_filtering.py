import argparse
import json
import warnings

import pandas as pd

from GGR.util.io.format_funcs import load_json, save_json, split_dataframe, join_dataframe

warnings.filterwarnings("ignore")


def assign_viewpoint(viewpoint, excluded_viewpoints):
    """
    Assign or modify viewpoint values to "right" or "left".

    Parameters:
    - viewpoint: Current viewpoint value to be assigned or modified.
    - excluded_viewpoints: List of viewpoint values to be excluded.

    Returns:
    - Assigned or modified viewpoint value.
    """

    if viewpoint is None:
        return None
    if viewpoint in excluded_viewpoints:
        return None
    if "left" in viewpoint:
        return "left"
    elif "right" in viewpoint:
        return "right"
    else:
        return None


def assign_viewpoints(df, excluded_viewpoints):
    """
    Assign or modify viewpoint values in a DataFrame based on specified rules.

    Parameters:
    - df: DataFrame containing 'viewpoint' column to be modified.
    - excluded_viewpoints: List of viewpoint values to be excluded.

    Returns:
    - DataFrame with assigned or modified 'viewpoint' values, excluding rows with NaN in 'viewpoint'.
    """
    for index, row in df.iterrows():
        df.at[index, "viewpoint"] = assign_viewpoint(
            row["viewpoint"], excluded_viewpoints
        )

    # Filter out rows with NaN in the 'viewpoint' column
    df = df[~df["viewpoint"].isna()]
    return df


def convert_bbox(bbox_str):
    bbox_values = bbox_str.strip("[]").split(", ")
    return [float(value) for value in bbox_values]


if __name__ == "__main__":
    print("Loading data...")
    parser = argparse.ArgumentParser(
        description="Detect bounding boxes for database of animal images"
    )
    parser.add_argument(
        "csv_file", type=str, help="The path to the CSV file to convert."
    )
    parser.add_argument(
        "eda_out", type=str, help="The location to save the JSON preprocessed annots."
    )
    parser.add_argument(
        "--video", action="store_true", help="True if we are processing video data."
    )
    args = parser.parse_args()

    video_mode = args.video

    data = load_json(args.csv_file)
    df = join_dataframe(data)

    print("Filtering data...")
    # filter out for true CA annotations
    df = df[df["annotations_census"] == True]

    # Drop the annotation_census column
    df = df.drop("annotations_census", axis=1)

    # Make individual id column
    df["individual_id"] = 0

    print("Reassigning viewpoints...")
    # Reassign all viewpoints to just left/right
    df = assign_viewpoints(df, excluded_viewpoints=["upback", "upfront"])

    print("Saving data...")
    # Save data
    annotations = split_dataframe(df)
    save_json(annotations, args.eda_out)

    print("Data is saved to:", args.eda_out)
