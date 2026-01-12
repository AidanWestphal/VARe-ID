import numpy as np
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
import os
import yaml


with open('config_visualize.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
IMG_DIR = config["img_dir"]
ANNOTS_DIR = config["annots_dir"]
FIELDS = config['desired_fields']
FULL_IMAGE = config['full_image']
SORT_BY = config['sort_by']
THRESHOLD = config['threshold']
PRINT = config['print_below']
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

# GET THE ANNOTATIONS CORRESPONDING TO EACH URI
images = uri_list
images_df = pd.DataFrame(images, columns=["uri"])
images_df["image_uuid"] = images_df["uri"].map(uri_uuid_mapping)

with open(ANNOTS_DIR, "r") as f:
    data = json.load(f)
    annot_df = pd.DataFrame(data["annotations"])

# Inner join because we can process this via a select w/ no returns
df = pd.merge(images_df, annot_df, on="image_uuid", how="inner")
if SORT_BY is not None:
    df = df.sort_values(by=SORT_BY)

if ANNOT_METHOD == "all":
    pass  # keep all rows

elif IMAGE_PATHS is not None:
    df = df[df["uri"].isin(IMAGE_PATHS)]

elif NUM_IMAGES is not None:
    num = min(NUM_IMAGES, len(df))
    df = df.sample(n=num)

else:
    raise Exception("Invalid inputs. Must specify either num_images, image_paths, or all.")


print(len(df))




from matplotlib.backends.backend_agg import FigureCanvasAgg

# Keep a fixed ordered list of URIs
URI_LIST = list(df["uri"].unique())
NUM_IMAGES = len(URI_LIST)

def visualize_image_by_index(idx):
    uri = URI_LIST[idx]
    select = df[df["uri"] == uri]

    img = cv2.imread(uri)
    annotation_text = ""

    if select.size > 0:
        for _, row in select.iterrows():
            bbox = np.array(row["bbox"]).astype(int)

            # ---- collect stats text ALWAYS ----
            annotation_text += f"image_uuid: {row['image_uuid']}\n"
            for index in FIELDS:
                annotation_text += f"{index}: {row[index]}\n"
            annotation_text += "\n"

            # ---- color logic ----
            if THRESHOLD == 0:
                color_num = np.random.randint(0, 179)
                hsv_color = np.uint8([[[color_num, 255, 255]]])
                color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB).flatten().tolist()
            else:
                if row[SORT_BY] > THRESHOLD:
                    color = np.array((0,255,0)).tolist()
                else:
                    color = np.array((0,0,255)).tolist()

            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            # ---------- CROPPED MODE ----------
            if not FULL_IMAGE:
                img_cropped = img[y1:y2, x1:x2]
                img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
                return img_cropped, annotation_text

            # ---------- FULL IMAGE MODE ----------
            img = cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness=10)

            above, below = 0, 1
            for index in FIELDS:
                annot_str = (
                    f"{index}: {row[index]:.2f}"
                    if isinstance(row[index], float)
                    else f"{index}: {row[index]}"
                )

                (tw, th), _ = cv2.getTextSize(
                    annot_str,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    img.shape[1] // 460
                )

                x_shift = max(0, x1 + tw - img.shape[1])
                gray_color = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                text_color = [255,255,255] if gray_color < 127.5 else [0,0,0]

                if 0 < y1 - th - 2 - above*80 < img.shape[0]:
                    img = cv2.rectangle(
                        img,
                        (x1-x_shift, y1 - th - 2 - above*80),
                        (x1-x_shift + tw + 2, y1 - above*80),
                        color,
                        -1
                    )
                    img = cv2.putText(
                        img,
                        annot_str,
                        (x1-x_shift-1, y1+1-above*80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        text_color,
                        img.shape[1] // 500
                    )
                    above += 1
                else:
                    img = cv2.rectangle(
                        img,
                        (x1-x_shift, y2 - th - 2 + below*80),
                        (x1-x_shift + tw + 2, y2 + below*80),
                        color,
                        -1
                    )
                    img = cv2.putText(
                        img,
                        annot_str,
                        (x1-x_shift-1, y2+1+below*80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        text_color,
                        img.shape[1] // 500
                    )
                    below += 1

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, annotation_text


#######################################################################


import gradio as gr
from collections import defaultdict


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
    uri = uris[state["idx"]]
    rows = grouped[uri]

    img = cv2.imread(uri)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    for i, row in rows.iterrows():
        bbox = np.array(row["bbox"]).astype(int)
        x1, y1, bw, bh = bbox
        x2, y2 = x1 + bw, y1 + bh

        # Initialize state
        if i not in state["box_states"][uri]:
            # Positive census = green, negative = blue
            state["box_states"][uri][i] = "green" if row["annotations_census"] else "blue"

        color = (0,255,0) if state["box_states"][uri][i] == "green" else (0,0,255)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), color, 4)

        # Annotation text
        y_offset = y1 - 10 if y1 > 20 else y2 + 30
        text = ", ".join(f"{f}:{row[f]}" for f in FIELDS)
        img = cv2.putText(
            img, text, (x1, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    # Draw missed detections
    for (x,y) in state["missed"][uri]:
        img = cv2.circle(img, (x,y), 8, (255,0,0), -1)

    return img

'''
def toggle_box(evt: gr.SelectData):
    uri = uris[state["idx"]]
    x, y = evt.index

    rows = grouped[uri]
    for i, row in rows.iterrows():
        x1,y1,w,h = row["bbox"]
        if x1 <= x <= x1+w and y1 <= y <= y1+h:
            cur = state["box_states"][uri][i]
            state["box_states"][uri][i] = "blue" if cur == "green" else "green"
            break

    return render_image()


def add_missed(evt: gr.SelectData):
    uri = uris[state["idx"]]
    state["missed"][uri].append(evt.index)
    return render_image()
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


def next_img():
    state["idx"] = min(state["idx"] + 1, len(uris)-1)
    return render_image()

def prev_img():
    state["idx"] = max(state["idx"] - 1, 0)
    return render_image()


def compute_stats():
    TP = FP = FN = 0

    for uri, boxes in state["box_states"].items():
        rows = grouped[uri]
        for i, final in boxes.items():
            started_green = rows.loc[i, "annotations_census"]
            if started_green and final == "green":
                TP += 1
            elif started_green and final == "blue":
                FP += 1
            elif (not started_green) and final == "green":
                FN += 1

        FN += len(state["missed"][uri])

    return f"TP: {TP}\nFP: {FP}\nFN: {FN}"


with gr.Blocks() as demo:
    img = gr.Image(label="Annotated Image", interactive=True)
    mode = gr.Radio(["Edit Boxes", "Missed Detection"], value="Edit Boxes")

    with gr.Row():
        prev_btn = gr.Button("Previous")
        next_btn = gr.Button("Next")

    stats_btn = gr.Button("Compute Stats")
    stats_out = gr.Textbox()

    img.select(
        fn=handle_click,
        inputs=mode,
        outputs=img
    )


    prev_btn.click(prev_img, outputs=img)
    next_btn.click(next_img, outputs=img)
    stats_btn.click(compute_stats, outputs=stats_out)

    demo.load(render_image, outputs=img)

demo.launch(share=True)



#######################################################################
'''

import gradio as gr

def init_state():
    return {
        "idx": 0,
        "labels": [None] * NUM_IMAGES
    }

def load_current(state):
    img, annot_text = visualize_image_by_index(state["idx"])
    status = f"Image {state['idx'] + 1} / {NUM_IMAGES}"

    label = state["labels"][state["idx"]]
    verdict = "Not reviewed" if label is None else ("❌ Error" if label else "✅ Correct")

    return img, annot_text, status, verdict, state


def mark_error(state):
    state["labels"][state["idx"]] = True
    return load_current(state)

def mark_correct(state):
    state["labels"][state["idx"]] = False
    return load_current(state)

def next_image(state):
    if state["idx"] < NUM_IMAGES - 1:
        state["idx"] += 1
    return load_current(state)

def prev_image(state):
    if state["idx"] > 0:
        state["idx"] -= 1
    return load_current(state)

def finish(state):
    error_uuids = []

    for i, v in enumerate(state["labels"]):
        if v is True:
            uri = URI_LIST[i]
            uuid = df[df["uri"] == uri]["image_uuid"].iloc[0]
            error_uuids.append(uuid)

    summary = (
        f"Total images: {NUM_IMAGES}\n"
        f"Errors: {len(error_uuids)}\n"
        f"Correct: {state['labels'].count(False)}\n"
        f"Unreviewed: {state['labels'].count(None)}\n\n"
        f"Image UUIDs with errors:\n{error_uuids}"
    )

    return summary



with gr.Blocks() as demo:
    gr.Markdown("## Annotation Error Review Tool")

    state = gr.State(init_state())

    with gr.Row():
        image = gr.Image(label="Annotated Image")
        annotation_box = gr.Textbox(
            label="Annotation Details",
            lines=15,
            interactive=False
        )

    status = gr.Textbox(label="Progress")
    verdict = gr.Textbox(label="Current Label")

    with gr.Row():
        prev_btn = gr.Button("⬅️ Previous")
        next_btn = gr.Button("➡️ Next")

    with gr.Row():
        correct_btn = gr.Button("✅ Correct")
        error_btn = gr.Button("❌ Error")

    finish_btn = gr.Button("Finish Review")
    summary = gr.Textbox(label="Summary", lines=10)

    demo.load(
        load_current,
        state,
        [image, annotation_box, status, verdict, state]
    )

    prev_btn.click(
        prev_image,
        state,
        [image, annotation_box, status, verdict, state]
    )
    next_btn.click(
        next_image,
        state,
        [image, annotation_box, status, verdict, state]
    )

    correct_btn.click(
        mark_correct,
        state,
        [image, annotation_box, status, verdict, state]
    )
    error_btn.click(
        mark_error,
        state,
        [image, annotation_box, status, verdict, state]
    )

    finish_btn.click(finish, state, summary)

demo.launch(share=True)
'''