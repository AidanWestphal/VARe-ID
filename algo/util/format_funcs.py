import json
import os
import yaml

import pandas as pd

# DESIRED COLUMNS TO BE KEPT WITHIN ANNOTATIONS
ANNOTATION_COLNAMES = [
    "uuid",
    "image_uuid",
    "video_path",
    "bbox",
    "viewpoint",
    "tracking_id",
    "individual_id",
    "confidence",
    "detection_class",
    "annotations_census",
    "CA_score",
    "category_id",
    "LCA_clustering_id"
]

# COLUMNS TO BE KEPT IN CATEGORIES (as found in annotations)
CATEGORY_COLNAMES = [
    "category_id",
    "species"
]

# RENAME PATTERNS FOR CATEGORIES (to be converted from above)
# NOTE: Renaming patterns should be REVERSIBLE
CATEGORY_COL_RENAMES = {
    "category_id": "id"
}

# COLUMNS TO BE KEPT IN IMAGES (as found in annotations)
IMAGE_COLNAMES = [
    "image_uuid",
    "image_path",
    "video_path",
    "timestamp",
    "frame_number"
]

# RENAME PATTERNS FOR IMAGES (to be converted from above)
# NOTE: Renaming patterns should be REVERSIBLE
IMAGE_COL_RENAMES = {
    "image_uuid": "uuid"
}


def load_config(file_path):
    '''
    Load a config file (.yaml) from a given path.
    '''
    with open(file_path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file


def load_json(file_path):
    '''
    Load a file (.json) from a given path.
    '''
    with open(file_path, "r") as file:
        return json.load(file)
    

def save_json(file, file_path):
    '''
    Save a file in record format (list of dictionaries) to JSON.
    '''
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(file, json_file, indent=4)
    

def load_csv(file_path):
    '''
    Load a file (.csv) from a given path.

    Loads data in a record format (list of dictionaries).
    '''
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")


def save_csv(file, file_path):
    '''
    Save a file in record format (list of dictionaries) to CSV.
    '''
    df = pd.DataFrame(file)
    df.to_csv(file_path, index=False)


def split_dataframe(df):
    '''
    Reads in pandas dataframe format of annotations. Splits the 
    dataframe into three segments (categories, images, and annotations).
    The fields for each of these categories are described at the top 
    of this file. 

    Returns a record-based annotations json object like the following:
    {
        categories: [{}]
        images: [{}, {}]
        annotations: [{}, {}, {}, {}]
    }
    '''

    # OBTAIN THE IMAGES SECTION
    image_cols = df.columns.intersection(IMAGE_COLNAMES)
    df_images = (
        df[image_cols].drop_duplicates(keep="first").reset_index(drop=True)
    )

    # PERFORM RENAMING
    image_renames = {key: value for key, value in IMAGE_COL_RENAMES.items() if key in df_images.columns}
    df_images = df_images.rename(columns=image_renames)

    # OBTAIN THE CATEGORY SECTION
    category_cols = df.columns.intersection(CATEGORY_COLNAMES)
    df_categories = (
        df[category_cols].drop_duplicates(keep="first").reset_index(drop=True)
    )

    # PERFORM RENAMING
    category_renames = {key: value for key, value in CATEGORY_COL_RENAMES.items() if key in df_categories.columns}
    df_categories = df_categories.rename(columns=category_renames)

    # DROP UNNECESSARY COLUMNS FROM ANNOTATIONS
    annot_cols = df.columns.intersection(ANNOTATION_COLNAMES)
    df_annots = df[annot_cols]

    final_json = {
        "categories": df_categories.to_dict(orient="records"),
        "images": df_images.to_dict(orient="records"),
        "annotations": df_annots.to_dict(orient="records"),
    }

    return final_json


def join_dataframe(annots):
    '''
    Joins a record-based annotations json object into a pandas dataframe.

    Returns a single aggregate dataframe. This is the reverse of split_dataframe.
    '''

    # GET THE THREE ANNOTATION SECTIONS
    df_categories = pd.DataFrame(annots["categories"])
    df_images = pd.DataFrame(annots["images"])
    df_annots = pd.DataFrame(annots["annotations"])

    # UNDO COLUMN RENAMING
    image_renames = {value: key for key, value in IMAGE_COL_RENAMES.items() if value in df_images.columns}
    df_images = df_images.rename(columns=image_renames)

    category_renames = {value: key for key, value in CATEGORY_COL_RENAMES.items() if value in df_categories.columns}
    df_categories = df_categories.rename(columns=category_renames)

    # MERGE SECTIONS
    # NOTE: Only merge if there is SPLIT DATA, e.g. some data may not be generated yet (like categories)
    m1_criterion = list(df_annots.columns.intersection(df_images.columns))
    if m1_criterion:
        m1 = df_annots.merge(df_images, on=m1_criterion, how='left')
    else:
        m1 = df_annots
    
    m2_criterion = list(m1.columns.intersection(df_categories.columns))
    if m2_criterion:
        m2 = m1.merge(df_categories, on=m2_criterion, how='left')
    else:
        m2 = m1

    return m2
    


def join_dataframe_dict(annots):
    '''
    Joins a record-based annotations json object into a pandas dataframe.

    Returns a record-based annotations json object with the "annotations" 
    entry being replaced with the joined element. This function exists as 
    some functions may be heavily dependent on non-pandas-formatted data.
    '''

    # GET THE THREE ANNOTATION SECTIONS
    df_categories = pd.DataFrame(annots["categories"])
    df_images = pd.DataFrame(annots["images"])
    df_annots = pd.DataFrame(annots["annotations"])

    # UNDO COLUMN RENAMING
    image_renames = {value: key for key, value in IMAGE_COL_RENAMES.items() if value in df_images.columns}
    df_images = df_images.rename(columns=image_renames)

    category_renames = {value: key for key, value in CATEGORY_COL_RENAMES.items() if value in df_categories.columns}
    df_categories = df_categories.rename(columns=category_renames)

    # MERGE SECTIONS
    # NOTE: Only merge if there is SPLIT DATA, e.g. some data may not be generated yet (like categories)
    m1_criterion = list(df_annots.columns.intersection(df_images.columns))
    if m1_criterion:
        m1 = df_annots.merge(df_images, on=m1_criterion, how='left')
    else:
        m1 = df_annots
    
    m2_criterion = list(m1.columns.intersection(df_categories.columns))
    if m2_criterion:
        df_annots = m1.merge(df_categories, on=m2_criterion, how='left')
    else:
        df_annots = m1

    final_json = {
        "categories": df_categories.to_dict(orient="records"),
        "images": df_images.to_dict(orient="records"),
        "annotations": df_annots.to_dict(orient="records"),
    }

    return final_json


