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


with open('config_visualize.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
IMG_DIR = config["img_dir"]
ANNOTS_DIR = config["annots_dir"]
FIELDS = config['desired_fields']
FULL_IMAGE = config['full_image']
SORT_BY = config['sort_by']
VIDEO_MODE = config['data_video']
ANNOT_METHOD = config['annot_method']
IMAGE_PATHS = config['image_paths']
NUM_IMAGES = config['num_images']


with open(ANNOTS_DIR, 'r') as file:
    annots = json.load(file)

with open(IMG_DIR, 'r') as file:
    metadata = json.load(file)

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

# Step 2: Get the list of appropriate URIs to display

if ANNOT_METHOD == "all":
    images = uri_list

elif IMAGE_PATHS is not None:
    #df = df[df["uri"].isin(IMAGE_PATHS)]
    images_input = [path for path in IMAGE_PATHS if isinstance(path,str)]
    images = list(set(images_input) & set(uri_list))

elif NUM_IMAGES is not None:
    #num = min(NUM_IMAGES, len(df))
    #df = df.sample(n=num)
    rands = np.random.choice(len(uri_list), NUM_IMAGES, replace=False)
    images = list(np.array(uri_list)[rands])

else:
    raise Exception("Invalid inputs. Must specify either num_images, image_paths, or all.")


# GET THE ANNOTATIONS CORRESPONDING TO EACH URI
images_df = pd.DataFrame(images, columns=["uri"])

images_df["image_uuid"] = images_df["uri"].map(uri_uuid_mapping)

with open(ANNOTS_DIR, "r") as f:
    data = json.load(f)
    annot_df = pd.DataFrame(data["annotations"])

# Inner join because we can process this via a select w/ no returns
df = pd.merge(images_df, annot_df, on="image_uuid", how="inner")

if SORT_BY is not None:
    df = df.sort_values(by=SORT_BY)


# Keep a fixed ordered list of URIs
URI_LIST = list(df["uri"].unique())
NUM_IMAGES = len(URI_LIST)


grouped = {
    uri: df[df["uri"] == uri].reset_index(drop=True)
    for uri in df["uri"].unique()
}

uris = list(grouped.keys())


state = {
    "idx": 0,
    "box_states": defaultdict(dict),  # uri -> box_idx -> "green"/"blue"
    "missed": defaultdict(list),       # uri -> list of (x,y)
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
    header_color = (30, 30, 30)  # dark gray
    text_color = (255, 255, 255)

    img = cv2.rectangle(
        img,
        (0, 0),
        (w, header_height),
        header_color,
        -1
    )

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
        bbox = np.array(row["bbox"]).astype(int)
        x1, y1, bw, bh = bbox
        x2, y2 = x1 + bw, y1 + bh

        if i not in state["box_states"][uri]:
            state["box_states"][uri][i] = (
                "green" if row["annotations_census"] else "blue"
            )

        color = (0,255,0) if state["box_states"][uri][i] == "green" else (0,0,255)

        img = cv2.rectangle(img, (x1,y1), (x2,y2), color, 4)

        # Annotation text
        y_offset = max(y1 - 10, header_height + 25)
        text = ", ".join(f"{f}:{row[f]}" for f in FIELDS)

        img = cv2.putText(
            img,
            text,
            (x1, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
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
        started_green = bool(row["annotations_census"])
        final_state = state["box_states"][uri].get(i)

        if started_green and final_state == "green":
            TP += 1
        elif started_green and final_state == "blue":
            FP += 1
        elif (not started_green) and final_state == "green":
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



with gr.Blocks() as demo:
    header_display = gr.Markdown()
    
    img = gr.Image(interactive=True)

    


    mode = gr.Radio(["Edit Boxes", "Missed Detection"], value="Edit Boxes")

    stats_display = gr.Markdown()

    with gr.Row():
        prev_btn = gr.Button("Previous")
        next_btn = gr.Button("Next")

    stats_btn = gr.Button("Compute Stats")
    stats_out = gr.Markdown()

    demo.load(render_image, outputs=[img, header_display, stats_display])

    prev_btn.click(prev_img, outputs=[img, header_display, stats_display])
    next_btn.click(next_img, outputs=[img, header_display, stats_display])
    stats_btn.click(compute_global_metrics, outputs=stats_out)



    img.select(
        fn=handle_click,
        inputs=mode,
        outputs=[img, header_display, stats_display]
    )


demo.launch(share=True)
