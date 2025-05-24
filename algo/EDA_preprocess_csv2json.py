import argparse
import json

# import uuid
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
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data["annotations"])


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

    sep = args.csv_file.rfind("/")
    annot_dir = args.csv_file[:sep]
    anno_path_csv = args.csv_file[sep:].replace("/", "")
    video_mode = args.video

    df = load_annotations_from_json(args.csv_file)

    # df = pd.read_csv(
    #     annot_dir + "/" + anno_path_csv, delimiter=",", quotechar='"', engine="python"
    # )

    # annot uuid,image uuid,image fname,video path,frame number,bbox x,bbox y,bbox w,bbox h,
    # bbox pred score,category id,tracking id,timestamp,species_prediction,species_pred_score,species_pred_simple,
    # path,bbox_xywh,bbox_xyxy,predicted_viewpoint,CA_score,annotations_census

    # filter out for true CA annotations.
    df = df[df["annotations_census"] == True]

    # df = df.merge(
    #     df[["image_path", "tracking_id", "individual_id"]],
    #     on=["image_path", "tracking_id"],
    #     how="left",
    # )

    # num_annotations = len(df)
    # num_images = len(df["image_path"].unique())

    # print("Dataset Statistics:")
    # print(f"Number of annotations: {num_annotations}")
    # print(f"Number of images: {num_images}")

    # # IF UUIDS AREN'T ALREADY PROVIDED (GT FILES), ADD THEM
    # if "image uuid" not in df.columns or "uuid" not in df.columns:
    #     image_uuid_map = {
    #         image_path: str(uuid.uuid4()) for image_path in df["image_path"].unique()
    #     }
    #     # Add the UUIDs to the DataFrame
    #     df["image_uuid"] = df["image_path"].map(image_uuid_map)
    #     df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # df["individual_id"] = df["individual_id_x"]

    df = assign_viewpoints(df, excluded_viewpoints=["upback", "upfront"])

    # IF CATEGORY ID NOT PROVIDED (GT)
    if "category_id" not in df.columns:
        df["category_id"] = 0

    # # IF FRAME NUMBER NOT PROVIDED (GT)
    # if video_mode:
    #     if "frame_number" not in df.columns:
    #         df["frame_number"] = -1
    #         for index, row in df.iterrows():
    #             df.at[index, "frame_number"] = int(
    #                 row["file_name"].split("_")[-1].split(".")[0]
    #             )

    if video_mode:
        df_annotations_fields = [
            "uuid",
            "image_uuid",
            "individual_id",
            "bbox",
            "viewpoint",
            "tracking_id",
            "confidence",
            "detection_class",
            "species",
            "CA_score",
            "category_id",
            "frame_number",
            "timestamp",
            "file_name",
            "image_path",
        ]
    else:
        df_annotations_fields = [
            "uuid",
            "image_uuid",
            "individual_id",
            "bbox",
            "viewpoint",
            "tracking_id",
            "confidence",
            "detection_class",
            "species",
            "CA_score",
            "category_id",
            "timestamp",
            "image_path",
        ]

    df_annotations = df[df_annotations_fields]

    df_images_fields = ["image_uuid", "image_path"]
    df_images_fields = df.columns.intersection(df_images_fields)
    df_images = (
        df[df_images_fields].drop_duplicates(keep="first").reset_index(drop=True)
    )
    df_images = df_images.rename(columns={"image_uuid": "uuid"})

    df_categories_fields = ["category_id", "species"]
    df_categories = (
        df[df_categories_fields].drop_duplicates(keep="first").reset_index(drop=True)
    )
    df_categories = df_categories.rename(columns={"category_id": "id"})

    result_dict = {
        "categories": df_categories.to_dict(orient="records"),
        "images": df_images.to_dict(orient="records"),
        "annotations": df_annotations.to_dict(orient="records"),
    }

    with open(args.eda_out, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4)
        print("Data is saved to:", args.eda_out)
