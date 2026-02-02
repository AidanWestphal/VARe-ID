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

SEED = None
NUM_IMAGES = 250

for i, arg in enumerate(sys.argv[1]):
    if arg == '--seed':
        if i+1 >= len(sys.argv[1]):
            raise("flag seed expects integer")
        try:
            SEED = int(sys.argv[1][i+1])
        except:
            raise("flag seed expects integer")
    
    if arg == '--num_images':
        if i+1 >= len(sys.argv[1]):
            raise("flag num_images expects integer")
        try:
            NUM_IMAGES = int(sys.argv[1][i+1])
        except:
            raise("flag num_images expects integer")
                
if SEED is not None:
    random.seed(SEED)

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

DIR = config['data_dir_out']
IDR_DIR = os.path.join(DIR, "id_region/id_regions.json")
IAC_DIR = os.path.join(DIR, config['ia_dirname'], config['ia_out_file'])
FIELDS = ['viewpoint', 'annotations_census', 'CA_score']
VIDEO_MODE = config['data_video']

# Random selection of images
if VIDEO_MODE:
    with open(os.path.join(DIR, config['dt_dirname'], config['dt_image_out_file']), 'r') as file:
            # Use json.load() to convert the file content to a Python dictionary
            data = json.load(file)
else:
    with open(os.path.join(DIR, config['dt_dirname'], config['dt_video_out_file']), 'r') as file:
            # Use json.load() to convert the file content to a Python dictionary
            data = json.load(file)
        
subset = random.sample(data['images'], NUM_IMAGES)
subset_uuids = [thing['uuid'] for thing in subset]
with open(os.path.join(DIR, config["image_out_file"]), 'r') as file:
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
    "missed": defaultdict(list)
}


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
            field_texts.append(f"{f}:{val}")
            
        text = ", ".join(field_texts)

        img = cv2.putText(
            img,
            text,
            (x1, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            base_color,
            2,
            cv2.LINE_AA
        )
    # ============================
    # Missed detections
    # ============================
    for (x, y) in state["missed"][uri]:
        img = cv2.circle(img, (x, y), 8, (255, 0, 0), -1)

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
        state["missed"][uri].append((x,y))

    return render_image()

def undo_missed():
    uri = uris[state["idx"]]
    if state["missed"][uri]:
        state["missed"][uri].pop()
    return render_image()


def next_img():
    state["idx"] = min(state["idx"] + 1, len(uris)-1)
    return render_image()

def prev_img():
    state["idx"] = max(state["idx"] - 1, 0)
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

    FN += len(state["missed"][uri])

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

        records.append({
            "image_uuid": uuid,
            "TP": TP,
            "FP": FP,
            "FN": FN
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
        ["Edit Boxes", "Missed Detection"],
        value="Edit Boxes",
        scale=4
        )

        undo_btn = gr.Button(
            "↩ Undo",
            scale=1,        # makes it small
            min_width=90   # keeps compact size
        )
    

    stats_display = gr.Markdown()

    with gr.Row():
        prev_btn = gr.Button("Previous")
        next_btn = gr.Button("Next")

    stats_btn = gr.Button("Compute Stats")
    stats_out = gr.Markdown()

    save_btn = gr.Button("Save Per-Image Results CSV")
    file_output = gr.File()

    demo.load(render_image, outputs=[img, header_display, stats_display])

    undo_btn.click(undo_missed, outputs=[img, header_display, stats_display])

    prev_btn.click(prev_img, outputs=[img, header_display, stats_display])
    next_btn.click(next_img, outputs=[img, header_display, stats_display])
    stats_btn.click(compute_global_metrics, outputs=stats_out)
    save_btn.click(save_results, outputs=file_output)

    img.select(
        fn=handle_click,
        inputs=mode,
        outputs=[img, header_display, stats_display]
    )


demo.launch(share=True)
