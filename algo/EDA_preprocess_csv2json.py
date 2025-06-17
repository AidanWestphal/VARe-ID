import argparse
import json
import warnings

import pandas as pd

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


def load_annotations_from_json(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data


def save_annotations_to_json(annotations_dict, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(annotations_dict, json_file, indent=4)


def convert_bbox(bbox_str):
    bbox_values = bbox_str.strip("[]").split(", ")
    return [float(value) for value in bbox_values]


if __name__ == "__main__":
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

    data = load_annotations_from_json(args.csv_file)
    df = pd.DataFrame(data["annotations"])

    # filter out for true CA annotations
    df = df[df["annotations_census"] == True]

    # Reassign all viewpoints to just left/right
    df = assign_viewpoints(df, excluded_viewpoints=["upback", "upfront"])

    # IF CATEGORY ID NOT PROVIDED (GT)
    # if "category_id" not in df.columns:
    #     df["category_id"] = 0

    # # IF FRAME NUMBER NOT PROVIDED (GT)
    # if video_mode:
    #     if "frame_number" not in df.columns:
    #         df["frame_number"] = -1
    #         for index, row in df.iterrows():
    #             df.at[index, "frame_number"] = int(
    #                 row["file_name"].split("_")[-1].split(".")[0]
    #             )

    # WE NEED TO REGENREATE IMAGES AND CATEGORIES AS WE FILTERED OUT ROWS
    df_images = (
        df[["image_uuid", "image_path"]].drop_duplicates(keep="first").reset_index(drop=True)
    )
    df_images = df_images.rename(columns={"image_uuid": "uuid"})

    df_categories = (
        df[["category_id", "species"]].drop_duplicates(keep="first").reset_index(drop=True)
    )
    df_categories = df_categories.rename(columns={"category_id": "id"})

    final_json = {
        "categories": df_categories.to_dict(orient="records"),
        "images": df_images.to_dict(orient="records"),
        "annotations": df.to_dict(orient="records"),
    }
    save_annotations_to_json(final_json, args.eda_out)

    print("Data is saved to:", args.eda_out)
