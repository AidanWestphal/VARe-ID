import json
import os
import shutil
import cv2
import pandas as pd
import yaml
from tqdm import tqdm
from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.workflow_funcs import build_config

# --- CONFIGURATION ---
CLASS_ID = 0 
OUTPUT_DIR = ""
VIS_YAML_PATH = ""
CONFIG_YAML_PATH = ""
# ---------------------

def convert_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    ncx = cx / img_w
    ncy = cy / img_h
    nw = w / img_w
    nh = h / img_h
    return [ncx, ncy, nw, nh]

def main():
    # 1. Load Configs
    vis_config = load_config(VIS_YAML_PATH)
    config = build_config(load_config(CONFIG_YAML_PATH))
    
    # Locate Session File
    session_path = os.path.join(config['data_dir_out'], vis_config['save_folder'], vis_config['save_name'])
    if not os.path.exists(session_path):
        print(f"Error: Session file not found at {session_path}")
        return

    print(f"Loading session from: {session_path}")
    with open(session_path, 'r') as f:
        session = json.load(f)

    # 2. Reconstruct Dataframe EXACTLY like visualizer
    # This is critical for index alignment
    VIDEO_MODE = config['data_video']
    
    if VIDEO_MODE:
        with open(config['video_out_path'], 'r') as f: data = json.load(f)
        image_metadata = []
        [image_metadata.extend(video["frame data"]) for video in data["videos"]]
    else:
        with open(config['image_out_path'], 'r') as f: data = json.load(f)
        image_metadata = data['images']

    # Map URI to UUID
    uri_uuid_mapping = {img["uri_original"]: img["uuid"] for img in image_metadata}
    uri_list = [img["uri_original"] for img in image_metadata]

    # Load Annotations
    with open(config['idr_out_path'], "r") as f:
        idr_df = pd.DataFrame(json.load(f)["annotations"])
        idr_df["is_secondary"] = False

    with open(config['ia_out_path'], "r") as f:
        iac_df = pd.DataFrame(json.load(f)["annotations"])
        iac_df["is_secondary"] = True

    # Filter IAC (Secondary) to remove duplicates present in IDR
    primary_uuids = set(idr_df["uuid"])
    iac_df = iac_df[~iac_df["uuid"].isin(primary_uuids)]

    all_annots_df = pd.concat([idr_df, iac_df], ignore_index=True)

    # Create Base Images DataFrame
    images_df = pd.DataFrame(uri_list, columns=["uri"])
    images_df["image_uuid"] = images_df["uri"].map(uri_uuid_mapping)

    # Merge exactly as before
    df = pd.merge(images_df, all_annots_df, on="image_uuid", how="left")

    # 3. Group by URI to get row indices
    grouped = {
        uri: df[df["uri"] == uri].reset_index(drop=True)
        for uri in df["uri"].unique()
    }

    # 4. Prepare Output Directories
    images_dir = os.path.join(OUTPUT_DIR, "images")
    labels_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 5. Process Images
    # We export any image that has an entry in 'box_errors' (meaning we viewed it)
    # OR any image that has manually added boxes.
    uris_to_process = set(session.get("box_errors", {}).keys()) | set(session.get("added_boxes", {}).keys())
    
    print(f"Exporting {len(uris_to_process)} verified images...")

    for uri in tqdm(uris_to_process):
        if not os.path.exists(uri): continue
        
        # Load Image for dimensions
        img = cv2.imread(uri)
        if img is None: continue
        h_img, w_img, _ = img.shape

        # Get the dataframe rows for this image
        # These rows contain the original model predictions
        if uri not in grouped: continue
        rows = grouped[uri]
        
        final_bboxes = []

        # --- A. Process TRUE POSITIVES (Originals - Errors) ---
        # Get the error map for this specific image from session
        # Format: {"0": true, "2": true} -> means box index 0 and 2 are WRONG (FP)
        error_map = session["box_errors"].get(uri, {})
        # JSON keys are strings, convert to int
        error_map = {int(k): v for k, v in error_map.items()}

        for i, row in rows.iterrows():
            if not isinstance(row["bbox"], list): continue

            is_secondary = row.get("is_secondary", False)
            census_true = bool(row["annotations_census"])
            if is_secondary:
                continue

            is_error = error_map.get(i, False)

            if not is_error and census_true:
                # If NOT marked as error, it is a TRUE POSITIVE. Keep it.
                final_bboxes.append(row["bbox"])
                
            if is_error and not census_true:
                # If marked as error, it is a FALSE NEGATIVE. Keep it.
                final_bboxes.append(row["bbox"])

        # --- B. Process Manually Added Boxes ---
        added_boxes = session["added_boxes"].get(uri, [])
        for bbox in added_boxes:
            # bbox is [x, y, w, h]
            final_bboxes.append(bbox)
            
        

        # --- C. Write to YOLO Format ---
        img_name = os.path.basename(uri)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        
        with open(os.path.join(labels_dir, txt_name), "w") as f:
            for bbox in final_bboxes:
                yolo_box = convert_to_yolo(bbox, w_img, h_img)
                f.write(f"{CLASS_ID} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")

        # Copy image to dataset folder
        shutil.copy2(uri, os.path.join(images_dir, img_name))

    # 6. Create Dataset YAML
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images',
        'val': 'images',
        'nc': 1,
        'names': ['grevy']
    }
    with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_content, f)

    print(f"\n✅ Dataset generated at: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()