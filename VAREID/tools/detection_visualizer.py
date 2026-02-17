import numpy as np
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
import os
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
import gradio as gr
from collections import defaultdict
import random
import sys

from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.workflow_funcs import build_config

vis_config = load_config("VAREID/tools/detection_visualize.yaml")

SEED = vis_config['seed']
NUM_IMAGES = vis_config['number_of_images']

if SEED is not None:
    try:
        SEED = int(SEED)
        print(f"Seed is {SEED}")
    except TypeError:
        raise("flag seed expects integer")
    random.seed(SEED)
    
try:
    NUM_IMAGES = int(NUM_IMAGES)
except TypeError:
    raise("flag num_images expects integer")
                
    

config = build_config(load_config('config.yaml'))

DIR = config['data_dir_out']
IDR_DIR = config['idr_out_path']
IAC_DIR = config['ia_out_path']
FIELDS = ['category_id', 'viewpoint', 'annotations_census', 'CA_score', 'clarity_score']
VIDEO_MODE = config['data_video']
VIS_FOlDER = os.path.join(DIR, vis_config['save_folder'])
VIS_SAVE_PATH = os.path.join(VIS_FOlDER, vis_config['save_name'])

if not os.path.isdir(VIS_FOlDER):
    os.mkdir(VIS_FOlDER)

# Random selection of images
if VIDEO_MODE:
    with open(config['dt_video_out_path'], 'r') as file:
            # Use json.load() to convert the file content to a Python dictionary
            data = json.load(file)
else:
    with open(config['dt_image_out_path'], 'r') as file:
            # Use json.load() to convert the file content to a Python dictionary
            data = json.load(file)
        
subset = random.sample(data['images'], NUM_IMAGES)
subset_uuids = [thing['uuid'] for thing in subset]
with open(config["image_out_path"], 'r') as file:
    data = json.load(file)

new_subset = []
for thing in data['images']:
    if thing['uuid'] in subset_uuids:
        new_subset.append(thing)
data['images'] = new_subset
metadata = data

# Step 1: Get the list of all valid image uris from metadata file
if VIDEO_MODE:
    image_metadata = []
    [image_metadata.extend(video["frame data"]) for video in metadata["videos"]]
else:
    image_metadata = metadata["images"]


uri_list = []
uri_uuid_mapping = {}
for image in image_metadata:
    uri_list.append(image["uri_original"])
    uri_uuid_mapping[image["uri_original"]] = image["uuid"]

# GET THE ANNOTATIONS CORRESPONDING TO EACH URI
images_df = pd.DataFrame(uri_list, columns=["uri"])

images_df["image_uuid"] = images_df["uri"].map(uri_uuid_mapping)

with open(IDR_DIR, "r") as f:
    data = json.load(f)
    idr_annot_df = pd.DataFrame(data["annotations"])
    idr_annot_df["is_secondary"] = False  # Mark as Primary

with open(IAC_DIR, "r") as f:
    data = json.load(f)
    iac_annot_df = pd.DataFrame(data["annotations"])
    iac_annot_df["is_secondary"] = True   # Mark as Secondary

primary_uuids = set(idr_annot_df["uuid"])

iac_annot_df = iac_annot_df[~iac_annot_df["uuid"].isin(primary_uuids)]

all_annots_df = pd.concat([idr_annot_df, iac_annot_df], ignore_index=True)

# Inner join because we can process this via a select w/ no returns
df = pd.merge(images_df, all_annots_df, on="image_uuid", how="left")

# Keep a fixed ordered list of URIs
URI_LIST = list(df["uri"].unique())

grouped = {
    uri: df[df["uri"] == uri].reset_index(drop=True)
    for uri in df["uri"].unique()
}

uris = list(grouped.keys())

'''
state = {
    "idx": 0,
    "box_states": defaultdict(dict),  # uri -> box_idx -> "green"/"blue"
    "missed": defaultdict(list),       # uri -> list of (x,y)
}'''

state = {
    "idx": 0,
    "box_errors": defaultdict(dict),  # uri -> box_idx -> True/False
    "missed": defaultdict(list),
    "added_boxes": defaultdict(list), # uri -> list of [x, y, w, h]
    "temp_pt": None                   # Used for 2-click box drawing
}

def save_session():
    """Saves the current state (edits, index, missed detections, and SEED) to JSON."""
    serializable_state = {
        "idx": state["idx"],
        "box_errors": state["box_errors"],
        "missed": state["missed"],
        "added_boxes": state["added_boxes"],
        "seed": SEED 
    }
    
    with open(VIS_SAVE_PATH, 'w') as f:
        json.dump(serializable_state, f, indent=4)
    
    return f"✅ Session saved to {VIS_SAVE_PATH}"

def load_session():
    """Loads state from JSON if it exists."""
    if not os.path.exists(VIS_SAVE_PATH):
        print("No previous session found. Starting fresh.")
        return

    try:
        with open(VIS_SAVE_PATH, 'r') as f:
            loaded = json.load(f)

        # Check if the seed has changed
        saved_seed = loaded.get("seed", None)
        
        if saved_seed != SEED or saved_seed is None:
            print(f"⚠️ Seed changed from {saved_seed} to {SEED}. Resetting index to 0.")
            state["idx"] = 0
        else:
            state["idx"] = loaded.get("idx", 0)
        
        # Restore 'missed': Convert back to defaultdict(list)
        state["missed"] = defaultdict(list, loaded.get("missed", {}))
        
        # Restore 'added_boxes': Convert back to defaultdict(list)
        state["added_boxes"] = defaultdict(list, loaded.get("added_boxes", {}))
        
        # Restore 'box_errors': JSON converts integer keys to strings.
        raw_errors = loaded.get("box_errors", {})
        restored_errors = defaultdict(dict)
        
        # LOAD ALL DATA (preserves edits for images not in current subset)
        for uri, error_map in raw_errors.items():
            int_key_map = {int(k): v for k, v in error_map.items()}
            restored_errors[uri] = int_key_map
            
        state["box_errors"] = restored_errors
        print(f"Successfully loaded session from {VIS_SAVE_PATH}")
        
    except Exception as e:
        print(f"Error loading session: {e}")

# Automatically load session on startup
load_session()

def render_image():
    idx = state["idx"]
    uri = uris[idx]
    rows = grouped[uri]

    img = cv2.imread(uri)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    total = len(uris)
    h, w, _ = img.shape

    # ============================
    # 🔹 Draw image UUID header
    # ============================
    image_uuid = rows.loc[0, "image_uuid"]

    header_height = 60
    header_color = (30, 30, 30)
    text_color = (255, 255, 255)

    img = cv2.rectangle(img, (0, 0), (w, header_height), header_color, -1)

    cv2.putText(
        img,
        f"Image UUID: {image_uuid}",
        (15, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        text_color,
        2,
        cv2.LINE_AA
    )

    # ============================
    # Bounding boxes
    # ============================
    for i, row in rows.iterrows():
        # 1. Check validity
        if not isinstance(row["bbox"], list):
            continue

        # 2. Get flags
        is_secondary = row.get("is_secondary", False)
        census_true = bool(row["annotations_census"])

        # 3. FILTER: If secondary and census is True, DO NOT display.
        if is_secondary and census_true:
            continue

        bbox = np.array(row["bbox"]).astype(int)
        x1, y1, bw, bh = bbox
        x2, y2 = x1 + bw, y1 + bh

        # 4. Color Logic
        # Since 'img' was converted to RGB at the top of the function:
        # Yellow = (255, 255, 0), Green = (0, 255, 0), Red = (255, 0, 0)
        
        if is_secondary:
            # We know census is False here because we skipped True above
            base_color = (255, 255, 0) # Yellow for Secondary
        else:
            # Primary dataset logic remains the same
            base_color = (0, 255, 0) if census_true else (0, 0, 255)

        # Draw box
        img = cv2.rectangle(img, (x1,y1), (x2,y2), base_color, 4)

        # Draw Error X (Manual edits)
        if state["box_errors"][uri].get(i, False):
            size = 72
            thickness = 8
            cx, cy = x2 + 8, y1 + 8
            # Blue X
            cv2.line(img, (cx-size, cy-size), (cx+size, cy+size), (0,0,255), thickness)
            cv2.line(img, (cx-size, cy+size), (cx+size, cy-size), (0,0,255), thickness)

        # Draw Text
        y_offset = max(y1 - 10, header_height + 25)
        
        # Helper to safely get field text
        field_texts = []
        for f in FIELDS:
            val = row.get(f, "N/A")
            if pd.isna(val): val = "N/A"
            if type(val) == float:
                val = f"{val:.2f}"
                
            field_texts.append(f"{f}:{val}")
            
        text = ", ".join(field_texts)

        img = cv2.putText(
            img,
            text,
            (x1, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (base_color),
            2,
            cv2.LINE_AA
        )
    # ============================
    # Newly Added Boxes (Manual)
    # ============================
    for (x, y, bw, bh) in state["added_boxes"][uri]:
        # Draw Cyan Box for manually added boxes
        img = cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 255), 4)
        
        # Add label
        img = cv2.putText(
            img, "Added", (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
        )

    # ============================
    # Missed detections
    # ============================
    for (x, y) in state["missed"][uri]:
        img = cv2.circle(img, (x, y), 8, (255, 0, 0), -1)

    # ============================
    # Temporary Point (Drawing in progress)
    # ============================
    if state["temp_pt"] is not None:
        tx, ty = state["temp_pt"]
        img = cv2.circle(img, (tx, ty), 6, (0, 255, 255), -1) # Small cyan dot
        cv2.putText(img, "Click 2nd Point", (tx+10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    TP, FP, FN = compute_image_stats(uri)

    header_text = (
        f"### 🆔 Image UUID: `{image_uuid}`  \n"
        f"### 🖼️ Image: **{idx + 1} / {total}**"
    )

    stats_text = (
        f"### 📊 Per-Image Stats\n"
        f"- ✅ **TP:** {TP}\n"
        f"- ❌ **FP:** {FP}\n"
        f"- ⚠️ **FN:** {FN}"
    )

    return img, header_text, stats_text

'''
def handle_click(evt: gr.SelectData, mode):
    if evt.index is None:
        return render_image()

    # Guard against None events
    if evt is None:
        return render_image()

    uri = uris[state["idx"]]
    x, y = evt.index

    if mode == "Edit Boxes":
        rows = grouped[uri]
        for i, row in rows.iterrows():
            x1, y1, w, h = row["bbox"]
            if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                cur = state["box_states"][uri][i]
                state["box_states"][uri][i] = (
                    "blue" if cur == "green" else "green"
                )
                break

    elif mode == "Missed Detection":
        state["missed"][uri].append((x, y))

    return render_image()
'''
def handle_click(evt: gr.SelectData, mode):
    if evt is None:
        return render_image()

    uri = uris[state["idx"]]
    x, y = evt.index

    if mode == "Edit Boxes":
        # Clear temp point if switching modes mid-draw
        state["temp_pt"] = None 

        rows = grouped[uri]

        for i, row in rows.iterrows():
            # --- FIX: Skip rows with no bbox ---
            if not isinstance(row["bbox"], list):
                continue
            # -----------------------------------
            
            x1,y1,w,h = row["bbox"]
            if x1 <= x <= x1+w and y1 <= y <= y1+h:
                cur = state["box_errors"][uri].get(i, False)
                state["box_errors"][uri][i] = not cur
                break

    elif mode == "Missed Detection":
        state["temp_pt"] = None
        state["missed"][uri].append((x,y))

    elif mode == "Add Box":
        # 2-Click Logic
        if state["temp_pt"] is None:
            # First Click
            state["temp_pt"] = (x, y)
        else:
            # Second Click - Finalize Box
            x1, y1 = state["temp_pt"]
            
            # Calculate coordinates
            x_min, x_max = min(x1, x), max(x1, x)
            y_min, y_max = min(y1, y), max(y1, y)
            w, h = x_max - x_min, y_max - y_min
            
            # Only add if box has size
            if w > 5 and h > 5:
                state["added_boxes"][uri].append([x_min, y_min, w, h])
            
            # Reset temp point
            state["temp_pt"] = None

    return render_image()

def undo_action(mode):
    """Context-aware undo function."""
    uri = uris[state["idx"]]

    # If in the middle of drawing a box, cancel the drawing
    if state["temp_pt"] is not None:
        state["temp_pt"] = None
        return render_image()

    if mode == "Missed Detection":
        if state["missed"][uri]:
            state["missed"][uri].pop()
            
    elif mode == "Add Box":
        if state["added_boxes"][uri]:
            state["added_boxes"][uri].pop()
            
    # For Edit Boxes, typically undo is just clicking the box again.
    
    return render_image()


def next_img():
    state["idx"] = min(state["idx"] + 1, len(uris)-1)
    state["temp_pt"] = None # Reset partial drawings
    return render_image()

def prev_img():
    state["idx"] = max(state["idx"] - 1, 0)
    state["temp_pt"] = None # Reset partial drawings
    return render_image()


def compute_image_stats(uri):
    TP = FP = FN = 0
    rows = grouped[uri]

    for i, row in rows.iterrows():
        # --- FIX: Skip rows with no bbox ---
        if not isinstance(row["bbox"], list):
            continue
        # -----------------------------------

        started_green = bool(row["annotations_census"])
        is_error = state["box_errors"][uri].get(i, False)

        if started_green and not is_error:
            TP += 1
        elif started_green and is_error:
            FP += 1
        elif (not started_green) and is_error:
            FN += 1

    # Count points and full boxes as False Negatives
    FN += len(state["missed"][uri])
    FN += len(state["added_boxes"][uri])

    return TP, FP, FN


def compute_global_metrics():
    TP = FP = FN = 0

    for uri in uris:
        t, f, n = compute_image_stats(uri)
        TP += t
        FP += f
        FN += n

    eps = 1e-8  # numerical safety

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * TP / (2 * TP + FP + FN + eps)
    accuracy = TP / (TP + FP + FN + eps)

    return (
        f"## 📈 Overall Dataset Metrics\n\n"
        f"- ✅ **True Positives (TP):** {TP}\n"
        f"- ❌ **False Positives (FP):** {FP}\n"
        f"- ⚠️ **False Negatives (FN):** {FN}\n\n"
        f"### 📊 Performance\n"
        f"- **Accuracy:** {accuracy:.4f}\n"
        f"- **Precision:** {precision:.4f}\n"
        f"- **Recall:** {recall:.4f}\n"
        f"- **F1 Score:** {f1:.4f}"
    )


def save_results():
    records = []

    for uri in uris:
        rows = grouped[uri]
        uuid = rows.loc[0, "image_uuid"]

        TP, FP, FN = compute_image_stats(uri)
        
        # We can include added boxes count in CSV if desired
        added_count = len(state["added_boxes"][uri])

        records.append({
            "image_uuid": uuid,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Added_Boxes": added_count
        })

    df_out = pd.DataFrame(records)
    save_path = "annotation_results.csv"
    df_out.to_csv(save_path, index=False)

    return save_path



with gr.Blocks() as demo:
    
    header_display = gr.Markdown()
    
    img = gr.Image(interactive=True)

    with gr.Row():
        mode = gr.Radio(
            ["Edit Boxes", "Missed Detection", "Add Box"],
            value="Edit Boxes",
            label="Tool Mode",
            scale=4
        )

        undo_btn = gr.Button(
            "↩ Undo Last Action",
            scale=1,        # makes it small
            min_width=90   # keeps compact size
        )
    

    stats_display = gr.Markdown()

    with gr.Row():
        prev_btn = gr.Button("Previous")
        next_btn = gr.Button("Next")

    stats_btn = gr.Button("Compute Stats")
    stats_out = gr.Markdown()
    
    with gr.Row():
        save_session_btn = gr.Button("💾 Save Progress (Session)", variant="primary")
        save_csv_btn = gr.Button("Save Per-Image Results CSV")
        
    session_out = gr.Markdown() # To show the "Saved" message
    file_output = gr.File()

    file_output = gr.File()

    demo.load(render_image, outputs=[img, header_display, stats_display])

    # Pass 'mode' to undo so it knows what to undo
    undo_btn.click(undo_action, inputs=[mode], outputs=[img, header_display, stats_display])

    prev_btn.click(prev_img, outputs=[img, header_display, stats_display])
    next_btn.click(next_img, outputs=[img, header_display, stats_display])
    stats_btn.click(compute_global_metrics, outputs=stats_out)
    
    save_session_btn.click(save_session, outputs=session_out)
    save_csv_btn.click(save_results, outputs=file_output)

    img.select(
        fn=handle_click,
        inputs=mode,
        outputs=[img, header_display, stats_display]
    )


demo.launch(share=True)